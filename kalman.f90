subroutine kalman_filter(z_meas, A, H, Q, R, P, x_input, x_estim, p_o)
  real, intent(in) :: A, H, Q, R, P, x_input
  real, intent(in) :: z_meas(1)
  real, intent(out) :: x_estim(1), p_o
  real :: x_pred(1), p_pred, K
  
  x_pred = A * x_input
  p_pred = A * P * A + Q
  K = p_pred * H / (H * p_pred * H + R)
  x_estim = x_pred + K * (z_meas - H * x_pred)
  p_o = p_pred - K * H * p_pred

end subroutine kalman_filter
