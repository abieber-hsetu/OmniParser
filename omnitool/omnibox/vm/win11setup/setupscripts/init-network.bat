@echo off
powershell -Command "Get-NetAdapter | Set-DnsClientServerAddress -ServerAddresses ('8.8.8.8', '1.1.1.1')"
ipconfig /flushdns
timeout /t 2 /nobreak >nul
net use E: \\172.18.0.1\Data /persistent:yes