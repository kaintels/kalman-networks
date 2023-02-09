subroutine SimpleKalman(z_meas, A, H, Q, R, P, x_estim, p_o)
  real, intent(in) :: A, H, Q, R, P
  real, intent(in) :: z_meas(2, 2)
  real, intent(out) :: x_estim(2, 2), p_o
  real :: x_pred(2,2), p_pred, K
  
  x_pred = A * x_estim
  p_pred = A * P * A + Q
  K = p_pred * H / (H * p_pred * H * R)
  x_estim = x_pred + K * (z_meas - H * x_pred)
  p_o = p_pred - K * H* p_pred

end subroutine SimpleKalman
