// send user query to get bot response
$('#user_query').keypress(function(e) { 
    if (e.keyCode === 13) {
        e.preventDefault();

        // get user query and display to dialog history
        var user_query = $('#user_query').val().trim();
        if (user_query.length == 0) {
            alert('No message is typed.')
            return;
        }
        var dialog_history = document.querySelector(".dialog_history");
        dialog_history.innerHTML += `<p class="user_msg">${user_query}</p>`;
        $('.dialog_history').scrollTop($('.dialog_history')[0].scrollHeight);
        $('#user_query').val('');

        // get context from dialog history
        var context = [];
        var context_list = document.querySelectorAll('[class*="msg"]');
        for (var i = 0; i < context_list.length; i++) {
            context.push(context_list[i].innerText);
        }
        
        // call endpoint
        $.ajax({
            url: '/send',
            method: 'post',
            data: JSON.stringify({context: context}),
            dataType: 'json',
            contentType: 'application/json',
            success: function(res_data) {
                var dialog_history = document.querySelector(".dialog_history");
                dialog_history.innerHTML += `<p class="bot_msg">${res_data.bot_response}</p>`;
                if (res_data.image_url) {
                    dialog_history.innerHTML += `<img class=image src=${res_data.image_url}>`;
                }
                $('.dialog_history').scrollTop($('.dialog_history')[0].scrollHeight);
            }
        });
    }
});


// iterate through turn evaluations
var get_turn_evaluations = function() {
    var evaluations = [];
    $('[class*="msg"]').each(function(){
    //$("input[type='checkbox']").each(function(){
        var ischecked = $(this).is(":checked");
        evaluations.push({"metric": $(this).attr("id"), "statement": $(this).next("label").text(), "is_checked": ischecked});
    });
    return evaluations;
};


// iterate through session evaluations
var get_session_evaluations = function() {
    var evaluations = [];
    $("input[type='checkbox']").each(function(){
        var ischecked = $(this).is(":checked");
        evaluations.push({"metric": $(this).attr("id"), "statement": $(this).next("label").text(), "is_checked": ischecked});
    });
    return evaluations;
};


// save evaluation results
$('#save').click(function(e) {
    e.preventDefault();
    var button = $(this);
    button.prop('disabled', true);

    // get crowdworker name
    var name = $('#name').val().trim(); 
    if (name.length == 0) {
        alert('Type your name.')
	    return;
    }

    // get context from dialog history
    var context = [];
    var context_list = document.querySelectorAll('[class*="msg"]');
    for (var i = 0; i < context_list.length; i++) {
        context.push(context_list[i].innerText);
    }
    
    // get evaluations
    var evaluations = get_evaluations();

    // call endpoint
    $.ajax({
        url: '/save',
        method: 'post',
        data: JSON.stringify({
            context: context, 
            evaluations: evaluations,
            name: name
        }),
        dataType: 'json',
        contentType: 'application/json',
        success: function(response) {
            $('#result_div').removeClass('d-none');
            $('input[type=checkbox]').prop('checked', false);
            $('#turn_workload').text('Turns: '+response.workload);
            button.prop('disabled', false);
        }
    });
});


// finish dialog
$('#done').click(function(e) {
    e.preventDefault();
    var button = $(this);
    button.prop('disabled', true);

    // get crowdworker name
    var name = $('#name').val().trim(); 
    if (name.length == 0) {
        alert('Type your name.')
	    return;
    }

    // get context from dialog history
    var context = [];
    var context_list = document.querySelectorAll('[class*="msg"]');
    for (var i = 0; i < context_list.length; i++) {
        context.push(context_list[i].innerText);
    }

    // get evaluations
    var evaluations = get_evaluations();

    // call endpoint
    $.ajax({
        url: '/done',
        method: 'post',
        data: JSON.stringify({
            context: context, 
            evaluations: evaluations,
            name: name
        }),
        dataType: 'json',
        contentType: 'application/json',
        success: function(response) {
            $('#result_div').removeClass('d-none');
            $('input[type=checkbox]').prop('checked', false);
            $('#session_workload').text('Sessions: '+response.workload);
            var dialog_history = document.querySelector(".dialog_history");
            dialog_history.innerHTML = `<p class="bot_msg">Hi, how are you?</p>`;
            button.prop('disabled', false);
        }
    });
});