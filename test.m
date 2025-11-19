% Quadrotor dynamics + PD + Feedforward + gravity (control en el for, modelo limpio en función)
clc; clear; close all;

% Parámetros
g = 9.81;
m = 1.0;
I = diag([0.02,0.02,0.04]);

ts = 0.01; Tsim = 30;
t = 0:ts:Tsim; N = numel(t);

% Trayectoria (círculo en XY, hover Z=1)
Rref = 0.5; w = 0.3;
xd   = Rref*cos(w*t);
yd   = Rref*sin(w*t);
zd   = ones(size(t));

xdp  = -Rref*w*sin(w*t);
ydp  =  Rref*w*cos(w*t);
zdp  = zeros(size(t));

xd2p = -Rref*w^2*cos(w*t);
yd2p = -Rref*w^2*sin(w*t);
zd2p = zeros(size(t));

% Ganancias (PD exterior y PD actitud)
% Nota: uso e = pd - pos y ed = vd - vel (signo corregido)
Kp = diag([1.8, 1.8, 4.5]);
Kd = diag([1.1, 1.1, 2.2]);

Kp_att = diag([6.0, 6.0, 6.0]);
Kd_att = diag([1.2, 1.2, 1.2]);

% Límites
T_min = 0.0;  T_max = 30.0;
phi_max = deg2rad(35); theta_max = deg2rad(35);

% Estado inicial
X0 = zeros(12,1);   % [pos(3); vel(3); ang(3); omega(3)]

% Almacenamiento
X = zeros(N, numel(X0)); X(1,:) = X0';

% Perturbación externa (apagada)
Fdist_fun = @(tk)[0;0;0];

% Simulación (control fuera, modelo dentro)
for k = 1:N-1
    % Estado actual
    xk = X(k,:)';
    pos   = xk(1:3);
    vel   = xk(4:6);
    ang   = xk(7:9);
    omega = xk(10:12);

    % Referencias
    pd = [xd(k); yd(k); zd(k)];
    vd = [xdp(k); ydp(k); zdp(k)];
    ad = [xd2p(k); yd2p(k); zd2p(k)];
    Fdist = Fdist_fun(t(k));

    % --- Control de posición (PD + FF + gravedad) ---
    e  = pd - pos;         % signo corregido
    ed = vd - vel;

    % Fuerza deseada en mundo
    u_des = m*(ad + Kp*e + Kd*ed + [0;0;g]);
    T_des = norm(u_des);
    if T_des < 1e-9
        zB_des = [0;0;1];
    else
        zB_des = u_des / T_des;
    end

    % Orientación deseada (yaw=0)
    phi_d   = asin(-zB_des(2));
    theta_d = atan2(zB_des(1), zB_des(3));
    psi_d   = 0;

    % Saturación de actitud deseada
    phi_d   = max(min(phi_d,   phi_max),   -phi_max);
    theta_d = max(min(theta_d, theta_max), -theta_max);

    % --- Control de actitud (PD) ---
    eta     = ang;
    eta_d   = [phi_d; theta_d; psi_d];
    tau     = -Kp_att*(eta - eta_d) - Kd_att*omega;

    % Saturar thrust
    T_cmd = max(min(T_des, T_max), T_min);

    % Integración del modelo con entradas (thrust y tau) ya calculadas
    [~, xx] = ode45(@(tt,xx)quad_model_only(tt, xx, T_cmd, tau, m, I, g, Fdist), ...
                    [t(k) t(k+1)], xk);
    X(k+1,:) = xx(end,:)';
end

% Plots
figure;
plot3(X(:,1),X(:,2),X(:,3),'b','LineWidth',1.5); hold on;
plot3(xd,yd,zd,'r--','LineWidth',1.5);
grid on; axis equal;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
legend('Trayectoria','Referencia');
title('Trayectoria 3D');

figure;
subplot(3,1,1); plot(t,X(:,1)-xd,'r'); ylabel('e_x [m]'); grid on;
subplot(3,1,2); plot(t,X(:,2)-yd,'g'); ylabel('e_y [m]'); grid on;
subplot(3,1,3); plot(t,X(:,3)-zd,'b'); ylabel('e_z [m]'); xlabel('Tiempo [s]'); grid on;
sgtitle('Errores de posición');

% --- Modelo dinámico únicamente (usando R world->body y thrust_world = R' * [0;0;T]) ---
function dX = quad_model_only(~, X, T, tau, m, I, g, Fdist)
    % Estados
    pos   = X(1:3);
    vel   = X(4:6);
    ang   = X(7:9);       phi=ang(1); theta=ang(2); psi=ang(3);
    omega = X(10:12);

    % Matriz de rotación ZYX (world -> body) con tercera fila [-sth, cth*sph, cth*cph]
    cps = cos(psi); sps = sin(psi);
    cth = cos(theta); sth = sin(theta);
    cph = cos(phi);   sph = sin(phi);
    R = [ cps*cth,  cps*sth*sph - sps*cph,  cps*sth*cph + sps*sph;
          sps*cth,  sps*sth*sph + cps*cph,  sps*sth*cph - cps*sph;
          -sth   ,  cth*sph              ,  cth*cph               ];
    % Nota: R mapea mundo -> cuerpo

    % Proyección del thrust al mundo: R' * [0;0;T] = T * [-sth; cth*sph; cth*cph]
    F_thrust_world = (R') * [0;0;T];

    % Dinámica traslacional
    acc = (1/m)*(F_thrust_world + Fdist) - [0;0;g];

    % Cinemática ángulos de Euler
    Wmat = [1 sph*tan(theta) cph*tan(theta);
            0 cph            -sph;
            0 sph/cth        cph/cth];
    ang_dot = Wmat*omega;

    % Dinámica rotacional (cuerpo)
    omega_dot = I \ (tau - cross(omega, I*omega));

    % Derivadas
    dX = zeros(12,1);
    dX(1:3)   = vel;
    dX(4:6)   = acc;
    dX(7:9)   = ang_dot;
    dX(10:12) = omega_dot;
end
