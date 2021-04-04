/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

var mycondition = condition;  // these two variables are passed by the psiturk server process
var mycounterbalance = counterbalance;  // they tell you which condition you have been assigned to
// they are not used in the stroop code but may be useful to you

// All pages to be loaded
var pages = [
	"instructions/instruct-1.html",
	//"instructions/instruct-2.html",	
	"instructions/instruct-ready.html",
	"stage.html",
	"postquestionnaire.html"
];

psiTurk.preloadPages(pages);

var instructionPages = [ // add as a list as many pages as you like
	"instructions/instruct-1.html",
	//"instructions/instruct-2.html",	
	"instructions/instruct-ready.html"
];


/********************
* HTML manipulation
*
* All HTML files in the templates directory are requested
* from the server when the PsiTurk object is created above. We
* need code to get those pages from the PsiTurk object and
* insert them into the document.
*
********************/

/********************
* STROOP TEST       *
********************/

var StroopExperiment = function() {

	psiTurk.recordUnstructuredData("mode", mode);

	var wordon; // time word is presented
	var TOTALNUMTRIALS = 28; //3 visual bins; 110 images per bin	
	var TOTALDUMMY = 6; //total number of dummy trials
	var TUNEDUMMY = 0.5; //how many proportion of dummy images you want TUEDUMMY*TOTALDUMMY/TOTALNUMTRIALS*SELECTEDTOTAL
	var SELECTEDTOTAL = TOTALNUMTRIALS + TOTALDUMMY*TUNEDUMMY ;
	//var listening = false; //keep listening  

    //load text file
    var xhttp;
    var dataText = "empty";
    var data_imglist=[];	
	
	if (window.XMLHttpRequest) {
	    // code for modern browsers
	    xhttp = new XMLHttpRequest();
	  } else {
	    // code for IE6, IE5
	    xhttp = new ActiveXObject("Microsoft.XMLHTTP");
	  }

	xhttp.onreadystatechange = function() {
	if (this.readyState == 4 && this.status == 200) 
	{     

		// process classlabels to int array
		//console.log(mycounterbalance+1);
	  	dataText = this.responseText.split("\n");
	  	//console.log(dataText);

	  	for(var i=0; i<TOTALNUMTRIALS; i++) 
	  	{ 
	  		data_imglist.push( dataText[i]);	  		
	  	} 		

		for (var i=0; i<TOTALDUMMY*TUNEDUMMY; i++)
	  	{ 
	  		data_imglist.push( -i-1 );	  		
	  	} 	
	    
		var imagelist = [];
		var imageID = [];
		var trialindex =-1;

	    imagenum = _.range(0,TOTALNUMTRIALS+TOTALDUMMY*TUNEDUMMY);	    
	    imagenum = _.shuffle(imagenum);
	    imagenum = imagenum.slice(0,SELECTEDTOTAL);
	    //console.log(imagenum);

	    for (i = 0; i < SELECTEDTOTAL; i++) 
	    { 
	    	var ind = imagenum[i];
	    	if (typeof data_imglist[ind] === 'number')
	    	{
	    		ind = -data_imglist[ind];
	    		ind = ind%TOTALDUMMY+1;
	    		var imagename = "http://kreiman.hms.harvard.edu/mturk/mengmi/dummy/dummy" + ind + ".jpg";    	
	    		imagelist.push(imagename);
	    	}
	    	else
	    	{
	    		var imagename = "http://kreimanlab.com/mengmiMturkHost/VirtualHome/keyframe_VH_gravity_gif/" + data_imglist[ind] ;   
	    		imagename = imagename.replace(/ /g, '%20'); 	
	    		imagelist.push(imagename);
	    	}
		} 
		psiTurk.preloadImages(imagelist);

		// Stimuli for a basic Stroop experiment	
		psiTurk.recordUnstructuredData("condition", mycondition);
		psiTurk.recordUnstructuredData("counterbalance", mycounterbalance);
		
		var next = function() {
			if (imagelist.length===0) {
				finish();
			}
			else {
				imageID = imagelist[0];
				current_img = imagelist.shift();				
				trialindex = trialindex+1;		
				d3.select("#stim").html('<img src='+current_img+' alt="stimuli" style="width:100%">');
				wordon = new Date().getTime();	
	        
			}
		};

		var finish = function() {
		    //$("body").unbind("keydown", response_handler); // Unbind keys
		    currentview = new Questionnaire();
		};
		

		// Load the stage.html snippet into the body of the page
		psiTurk.showPage('stage.html');

		// Register the response handler that is defined above to handle any
		// key down events.
		//$("body").focus().keydown(response_handler);

		// Start the test; initialize everything
		next();
		document.getElementById("submittrial").addEventListener("click", mengmiClick);

	    function containsSpecialCharacters(str)
	    {
		    var regex = /[ !~@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/g;
			return regex.test(str);
		}

		function mengmiClick() 
		{
			var response = document.getElementById("response").value;
		    document.getElementById("response").value = "";

			if (response.length > 0 && !containsSpecialCharacters(response) && !(/\s/.test(response)))
			{
				//document.getElementById("demo").innerHTML = response;
				var rt = new Date().getTime() - wordon;
				//console.log(imageID);
				psiTurk.recordTrialData({'phase':"TEST",
		                                 'imageID':imageID, //image name presented                                
		                                 'response':response, //worker response for image name 
		                                 'hit':imagenum[trialindex], //index of image name
		                                 'counterbalance': mycounterbalance+1, //type of choices in that trial
		                                 'rt':rt, //response time		                                 
		                             	 'trial': trialindex+1} //trial index starting from 1
		                               );

			    next();
			}
		    else
		    {

		    	window.alert("Warning: please type in one word (no space and special characters) before submitting your response!");
		    }

		}//mengmiClick end of function call
	} //if check reading text status
	}; //xhttp.onreadystatechange

	xhttp.open("GET", "static/ImageSet/mturkSet_" + (mycounterbalance+1) + ".txt", true);
	//xhttp.open("GET", "static/ImageSet/mturkSet_" + (792) + ".txt", true);
	xhttp.send();  
};


/****************
* Questionnaire *
****************/

var Questionnaire = function() {

	var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";

	record_responses = function() {

		psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'submit'});

		$('select').each( function(i, val) {
			psiTurk.recordUnstructuredData(this.id, this.value);
		});

	};

	prompt_resubmit = function() {
		document.body.innerHTML = error_message;
		$("#resubmit").click(resubmit);
	};

	resubmit = function() {
		document.body.innerHTML = "<h1>Trying to resubmit...</h1>";
		reprompt = setTimeout(prompt_resubmit, 10000);

		psiTurk.saveData({
			success: function() {
			    clearInterval(reprompt);
				psiTurk.completeHIT();
			},
			error: prompt_resubmit
		});
	};

	// Load the questionnaire snippet
	psiTurk.showPage('postquestionnaire.html');
	psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'begin'});

	$("#next").click(function () {
	    record_responses();
	    psiTurk.saveData({
            success: function(){
            	psiTurk.completeHIT(); // when finished saving compute bonus, the quit
            },
            error: prompt_resubmit});
	});


};

// Task object to keep track of the current phase
var currentview;

/*******************
 * Run Task
 ******************/
$(window).load( function(){
    psiTurk.doInstructions(
    	instructionPages, // a list of pages you want to display in sequence
    	function() { currentview = new StroopExperiment(); } // what you want to do when you are done with instructions
    );
});