// Auto-login helper for admin debug interface
document.addEventListener('DOMContentLoaded', function() {
  // Create the login form
  const loginForm = document.createElement('form');
  loginForm.classList.add('container', 'mt-5');
  
  loginForm.innerHTML = `
    <div class="row justify-content-center">
      <div class="col-md-6">
        <div class="card">
          <div class="card-header bg-primary text-white">
            <h3 class="mb-0">تسجيل دخول المشرف</h3>
          </div>
          <div class="card-body">
            <div class="alert alert-info">
              يمكنك تسجيل الدخول باستخدام حساب المشرف: 
              <strong>username: test, password: admin123</strong>
            </div>
            
            <div class="mb-3">
              <label for="username" class="form-label">اسم المستخدم</label>
              <input type="text" class="form-control" id="username" value="test">
            </div>
            
            <div class="mb-3">
              <label for="password" class="form-label">كلمة المرور</label>
              <input type="password" class="form-control" id="password" value="admin123">
            </div>
            
            <button type="submit" class="btn btn-primary">تسجيل الدخول</button>
          </div>
        </div>
      </div>
    </div>
  `;
  
  // Add the form to the page
  document.body.prepend(loginForm);
  
  // Handle form submission
  loginForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    submitLogin(username, password);
  });
  
  // Auto-login button has been removed
  
  // Login function
  function submitLogin(username, password) {
    fetch('/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
    })
    .then(response => {
      if (response.url.includes('/admin_debug')) {
        // Successfully redirected to admin debug
        window.location.href = '/admin_debug';
      } else {
        return response.text().then(html => {
          if (html.includes('تم تسجيل الدخول بنجاح') || html.includes('success')) {
            // Login successful
            window.location.href = '/admin_debug';
          } else {
            alert('فشل تسجيل الدخول. يرجى التحقق من اسم المستخدم وكلمة المرور.');
          }
        });
      }
    })
    .catch(error => {
      console.error('Error:', error);
      alert('حدث خطأ أثناء تسجيل الدخول. يرجى المحاولة مرة أخرى.');
    });
  }
});