#!/bin/bash

SITES_AVAILABLE="/etc/nginx/sites-available"
SITES_ENABLED="/etc/nginx/sites-enabled"

sudo tee $SITES_AVAILABLE/waf_ngnix > /dev/null <<EOL
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOL

if [ -f $SITES_AVAILABLE/waf_ngnix ]; then
    echo "Configuration file created successfully in $SITES_AVAILABLE."
else
    echo "Failed to create the configuration file."
    exit 1
fi

if [ -L $SITES_ENABLED/default ]; then
    sudo rm $SITES_ENABLED/default
    echo "Old default symlink removed."
fi

sudo ln -s $SITES_AVAILABLE/waf_ngnix $SITES_ENABLED/default

if [ -L $SITES_ENABLED/default ]; then
    echo "New symlink created successfully in $SITES_ENABLED."
else
    echo "Failed to create the symlink."
    exit 1
fi

sudo systemctl restart nginx

if [ $? -eq 0 ]; then
    echo "Nginx restarted successfully."
else
    echo "Failed to restart Nginx."
    exit 1
fi
