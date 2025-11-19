% Quadrotor dynamics + PD + Feedforward + gravity (control in the loop, clean model in function)
clc; clear; close all;

% Parameters
g = 9.81;
m = 1.0;
I = diag([0.02,0.02,0.04]);

ts = 0.01; Tsim = 30;
t = (0:ts:Tsim)';   % column vector to avoid broadcasting in subtractions
N = numel(t);

% Reference trajectory (circle in XY, hover at Z=1)
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

% Gains (outer PD and attitude PD)
% Note: using e = real - desired and control applies the correct sign
Kp = diag([1.8, 1.8, 4.5]);
Kd = diag([1.1, 1.1, 2.2]);

Kp_att = diag([6.0, 6.0, 6.0]);
Kd_att = diag([1.2, 1.2, 1.2]);

% Limits
T_min = 0.0;  T_max = 30.0;
phi_max = deg2rad(35); theta_max = deg2rad(35);

% Initial state
X0 = zeros(12,1);   % [pos(3); vel(3); ang(3); omega(3)]

% Storage
X = zeros(N, numel(X0)); X(1,:) = X0';

% External disturbance (disabled)
Fdist_fun = @(tk)[0;0;0];

% Simulation (control outside, model inside)
for k = 1:N-1
    % Current state
    xk = X(k,:)';
    pos   = xk(1:3);
    vel   = xk(4:6);
    ang   = xk(7:9);
    omega = xk(10:12);

    % References
    pd = [xd(k); yd(k); zd(k)];
    vd = [xdp(k); ydp(k); zdp(k)];
    ad = [xd2p(k); yd2p(k); zd2p(k)];
    Fdist = Fdist_fun(t(k));

    % --- Position control (PD + FF + gravity) ---
    e  = pos - pd;         % real - desired
    ed = vel - vd;

    % Desired force in world frame
    u_des = m*(ad - Kp*e - Kd*ed + [0;0;g]);
    T_des = norm(u_des);
    if T_des < 1e-9
        zB_des = [0;0;1];
    else
        zB_des = u_des / T_des;
    end

    % Desired orientation (yaw=0) solving zB = [-sin(theta); cos(theta)*sin(phi); cos(theta)*cos(phi)]
    % -> theta = asin(-zBx) and phi = atan2(zBy, zBz)
    theta_d = asin(-zB_des(1));
    phi_d   = atan2(zB_des(2), zB_des(3));
    psi_d   = 0;

    % Attitude saturation
    phi_d   = max(min(phi_d,   phi_max),   -phi_max);
    theta_d = max(min(theta_d, theta_max), -theta_max);

    % --- Attitude control (PD) ---
    eta     = ang;
    eta_d   = [phi_d; theta_d; psi_d];
    tau     = -Kp_att*(eta - eta_d) - Kd_att*omega;

    % Thrust saturation
    T_cmd = max(min(T_des, T_max), T_min);

    % Model integration with inputs (thrust and tau) already computed
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
legend('Trajectory','Reference');
title('Quadrotor 3D trajectory');

figure;
subplot(3,1,1); plot(t,X(:,1)-xd,'r'); ylabel('e_x [m]'); grid on;
subplot(3,1,2); plot(t,X(:,2)-yd,'g'); ylabel('e_y [m]'); grid on;
subplot(3,1,3); plot(t,X(:,3)-zd,'b'); ylabel('e_z [m]'); xlabel('Time [s]'); grid on;
sgtitle('Position errors');

% --- Dynamic model only (using R world->body and thrust_world = R' * [0;0;T]) ---
function dX = quad_model_only(~, X, T, tau, m, I, g, Fdist)
    % States
    pos   = X(1:3);
    vel   = X(4:6);
    ang   = X(7:9);       phi=ang(1); theta=ang(2); psi=ang(3);
    omega = X(10:12);

    % Rotation matrix ZYX (world -> body) with third row [-sth, cth*sph, cth*cph]
    cps = cos(psi); sps = sin(psi);
    cth = cos(theta); sth = sin(theta);
    cph = cos(phi);   sph = sin(phi);
    R = [ cps*cth,  cps*sth*sph - sps*cph,  cps*sth*cph + sps*sph;
          sps*cth,  sps*sth*sph + cps*cph,  sps*sth*cph - cps*sph;
          -sth   ,  cth*sph              ,  cth*cph               ];
    % Note: R maps world -> body

    % Thrust projection to world: R' * [0;0;T] = T * [-sth; cth*sph; cth*cph]
    F_thrust_world = (R') * [0;0;T];

    % Translational dynamics
    acc = (1/m)*(F_thrust_world + Fdist) - [0;0;g];

    % Euler angles kinematics
    Wmat = [1 sph*tan(theta) cph*tan(theta);
            0 cph            -sph;
            0 sph/cth        cph/cth];
    ang_dot = Wmat*omega;

    % Rotational dynamics (body)
    omega_dot = I \ (tau - cross(omega, I*omega));

    % Derivatives
    dX = zeros(12,1);
    dX(1:3)   = vel;
    dX(4:6)   = acc;
    dX(7:9)   = ang_dot;
    dX(10:12) = omega_dot;
end
