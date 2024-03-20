const sql = require('mssql');

const config = {
        user: 'Ayala',
        password: 'isr1953',
        server: '192.168.16.3',
        database: 'electric_ML',
        options: {
          encrypt: false, 
        },
      };

async function readDataDB(profiles = 0) {
    try {
        // Connect to the database
        await sql.connect(config);

        // Query to select data from your table
        const result = await sql.query`exec dbo.GetElectricPointChargeDetails_acc @amper2levels = ${profiles}`;

        // Map the result to an array of objects
        const data = result.recordset;

        console.log("reading from db");

        return data;
    } catch (err) {
        console.error('Error reading data from SQL Server:', err.message);
        throw err;
    } finally {
        await sql.close();
    }
}

async function saveToDb(key, xml, profiles = 0) {
    try {
        await sql.connect(config);

        const transaction = new sql.Transaction();
        await transaction.begin();

        const request = new sql.Request(transaction);

        let moduleCode = 0
        if (profiles == 0)
            moduleCode = 5
        else 
            moduleCode = 6


        //moduleCode 7
        
        await request.query`exec UpsertModels 7,${key},${xml}`;

        await transaction.commit();

        console.log(`Successfully saving ${key}`);
    } catch (err) {
        console.error('Error saving data to SQL Server:', err.message);
        throw err;
    } finally {
        await sql.close();
    }
}

module.exports = {
    readDataDB,
    saveToDb,
};
