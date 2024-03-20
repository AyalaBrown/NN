const { readDataDB } = require('./readingFromDB.js');
const { runModel } = require('./model.js');
const fs = require('fs');
const path = require('path');

const daysOfWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const currentDay = new Date().getDay();

try{
    if (process.argv.length > 2) {
        const profiles = process.argv[2];
        console.log(`Profiles: ${profiles}`);
        const logFileName = `${daysOfWeek[currentDay]}${profiles}.log`;

        // Define the directory for log files
        const logDirectory = path.join(__dirname, 'logs');

        // Ensure that the log directory exists
        if (!fs.existsSync(logDirectory)) {
            fs.mkdirSync(logDirectory);
        }

        // Define the path to the current day's log file
        const logFilePath = path.join(logDirectory, logFileName);

        // Create a write stream to the log file
        const logStream = fs.createWriteStream(logFilePath, { flags: 'w' }); // Use 'w' flag to override existing file

        // Redirect console.log to the log file
        const originalLog = console.log;

        console.log = function (message) {
            const formattedMessage = `[${new Date().toISOString()}] ${message}\n`;
            logStream.write(formattedMessage);
            originalLog.apply(console, arguments);
        };

        const originalError = console.error;
        console.error = function (error) {
            const formattedError = `[${new Date().toISOString()}] ERROR: ${error.stack}\n`;
            logStream.write(formattedError);
            originalError.apply(console, arguments);
        };

        run(profiles)
    } else {
        throw new Error("No profile provided");
    }
} catch(err){
    console.error(err)
}

async function run(profiles){
    const data = await readDataDB(profiles);
    return await runModel(data, profiles);
}
