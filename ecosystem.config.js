module.exports = {
  apps : [
     // Worker instance
     {
       name      : 'ml-worker',
       script    : 'scripts/celery-worker.sh',
       env: {
         DISPLAY: '127.0.0.1:0.0',
         BROKER_URL: 'redis://127.0.0.1:6379/0',
         CELERY_RESULT_BACKEND: `mongodb://${process.env.MONGO_HOST || "mongo"}:${process.env.MONGO_PORT || 27017}/celery`,
         GPU_MEM_USE: '0.3'
       },
       env_production : {
         DISPLAY: '127.0.0.1:0.0',
         CELERY_RESULT_BACKEND: `mongodb://${process.env.MONGO_HOST || "mongo"}:${process.env.MONGO_PORT || 27017}/celery`,
         BROKER_URL: 'redis://127.0.0.1:6379/0'
       }
     },

     // Server instance
     {
       name      : 'ml-server',
       script    : 'scripts/server.sh',
       env: {
         BROKER_URL: 'redis://127.0.0.1:6379/0',
         CELERY_RESULT_BACKEND: `mongodb://${process.env.MONGO_HOST || "mongo"}:${process.env.MONGO_PORT || 27017}/celery`
       },
       env_production : {
         NODE_ENV: 'production',
         BROKER_URL: 'redis://127.0.0.1:6379/0',
         CELERY_RESULT_BACKEND: `mongodb://${process.env.MONGO_HOST || "mongo"}:${process.env.MONGO_PORT || 27017}/celery`
       }
     },

    // Monitor celery instance
    {
      name      : 'ml-monitor',
      script    : 'scripts/monitor.sh',
      env: {
        BROKER_URL: 'redis://127.0.0.1:6379/0',
         CELERY_RESULT_BACKEND: `mongodb://${process.env.MONGO_HOST || "mongo"}:${process.env.MONGO_PORT || 27017}/celery`
      },
      env_production : {
        BROKER_URL: 'redis://127.0.0.1:6379/0',
         CELERY_RESULT_BACKEND: `mongodb://${process.env.MONGO_HOST || "mongo"}:${process.env.MONGO_PORT || 27017}/celery`
      }
    }
  ]
};
