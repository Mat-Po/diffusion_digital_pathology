function createFirstTaskQuestion(question, form) {
    var image_item = form.addImageItem()

    image_item.setTitle("Tiles id " + question[5])
    image_item.setImage(DriveApp.getFileById(question[5]).getBlob())

    var item = form.addMultipleChoiceItem();

    item.setTitle('Guess the tissue of the tiles above (id ' + question[5] + ')');
    item.setChoices([
        item.createChoice('Brain', question[4] == 'Brain'),
        item.createChoice('Lung', question[4] == 'Lung'),
        item.createChoice('Uterus', question[4] == 'Uterus'),
        item.createChoice('Kidney', question[4] == 'Kidney'),
        item.createChoice('Pancreas', question[4] == 'Pancreas')
    ])
    item.setPoints(1)

}

function createSecondTaskQuestion(question, form) {
    var image_item = form.addImageItem()

    image_item.setTitle("Tile id " + question[5])
    image_item.setImage(DriveApp.getFileById(question[5]).getBlob())

    var item = form.addMultipleChoiceItem();

    item.setTitle('Guess the origin of the tile above (id ' + question[5] + ')');
    item.setChoices([
        item.createChoice('real', question[3] == "real"),
        item.createChoice('fake', question[3] == "fake"),

    ])
    item.setPoints(1)

}


function getTileFilesMetadataFromFolder(folder) {
    var toRtn = []
    var files = folder.getFiles();
    var count = 1
    while (files.hasNext()) {
        file = files.next();
        var row = []
        var name = file.getName()
        var splits = name.split("-")
        var task = splits[0] == "first_task" ? "first" : "second"
        var tissue = task == "first" ? splits[1] : splits[0].split("_")[0]
        var isReal = "real"

        if (task == "first") {
            if (splits[2] == "fake") isReal = "fake"
        }
        else {
            if (splits.length < 10) isReal = "fake"
        }

        row.push(count, task, name, isReal, tissue, file.getId(), file.getSize(), file.getUrl())
        count = count + 1
        toRtn.push(row);
    }
    return toRtn
}

function build_questionnaires() {

    //questions that are the same for all questionnaires

    //var fixed_questions_folder = DriveApp.getFolderById(""); // I change the folder ID  here 
    //var fixed_questions_metadata = getTileFilesMetadataFromFolder(fixed_questions_folder)


    var questionaires_specific_questions_folders_id = [""]
    var number_questionaire = questionaires_specific_questions_folders_id.length

    questionaires_specific_questions_folders_id.forEach(function (questionnaire_folders_id) {
        console.log("DOING QUESTIONAIRE " + number_questionaire)
        var questionnaire_specific_folder = DriveApp.getFolderById(questionnaire_folders_id); // I change the folder ID  here 

        var qustionnaire_specific_questions = getTileFilesMetadataFromFolder(questionnaire_specific_folder)

        var questionnaire_questions = qustionnaire_specific_questions //fixed_questions_metadata.concat(qustionnaire_specific_questions)
        questionnaire_questions.sort(function (a, b) { return a[1].localeCompare(b[1]); });
        
        var sheet = SpreadsheetApp.openById("").getSheetByName("Sheet1");
        sheet.getRange(1,1,questionnaire_questions.length,8).setValues(questionnaire_questions);
        var form = FormApp.create('Questionnaire ' + number_questionaire);
        form.setIsQuiz(true)

        var question_number = 1
        questionnaire_questions.forEach(function (question) {

            var task = question[1]

            
            if (task == "first")
                createFirstTaskQuestion(question, form)
            else
                createSecondTaskQuestion(question, form)
            question_number = question_number + 1
        })



        number_questionaire = number_questionaire + 1
        // code
    });

}

function myFunction(){
  build_questionnaires()
}