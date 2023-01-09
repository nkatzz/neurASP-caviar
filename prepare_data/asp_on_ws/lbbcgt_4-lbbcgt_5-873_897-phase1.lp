
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lbbcgt_4,142,873). coords(lbbcgt_4,8,70,873).  visible(lbbcgt_4,873). happensAt(walking(lbbcgt_4),873). orientation(lbbcgt_5,103,873). coords(lbbcgt_5,252,285,873).  happensAt(appear(lbbcgt_5),873). happensAt(walking(lbbcgt_5),873).
orientation(lbbcgt_4,142,874). coords(lbbcgt_4,8,69,874).  visible(lbbcgt_4,874). happensAt(walking(lbbcgt_4),874). orientation(lbbcgt_5,100,874). coords(lbbcgt_5,253,284,874).  visible(lbbcgt_5,874). happensAt(walking(lbbcgt_5),874).
orientation(lbbcgt_4,142,875). coords(lbbcgt_4,8,67,875).  visible(lbbcgt_4,875). happensAt(walking(lbbcgt_4),875). orientation(lbbcgt_5,103,875). coords(lbbcgt_5,253,282,875).  visible(lbbcgt_5,875). happensAt(walking(lbbcgt_5),875).
orientation(lbbcgt_4,142,876). coords(lbbcgt_4,8,66,876).  visible(lbbcgt_4,876). happensAt(walking(lbbcgt_4),876). orientation(lbbcgt_5,103,876). coords(lbbcgt_5,253,282,876).  visible(lbbcgt_5,876). happensAt(walking(lbbcgt_5),876).
orientation(lbbcgt_4,137,877). coords(lbbcgt_4,8,63,877).  visible(lbbcgt_4,877). happensAt(walking(lbbcgt_4),877). orientation(lbbcgt_5,103,877). coords(lbbcgt_5,252,281,877).  visible(lbbcgt_5,877). happensAt(walking(lbbcgt_5),877).
orientation(lbbcgt_4,137,878). coords(lbbcgt_4,9,63,878).  visible(lbbcgt_4,878). happensAt(walking(lbbcgt_4),878). orientation(lbbcgt_5,103,878). coords(lbbcgt_5,252,281,878).  visible(lbbcgt_5,878). happensAt(walking(lbbcgt_5),878).
orientation(lbbcgt_4,137,879). coords(lbbcgt_4,9,63,879).  visible(lbbcgt_4,879). happensAt(walking(lbbcgt_4),879). orientation(lbbcgt_5,64,879). coords(lbbcgt_5,251,281,879).  visible(lbbcgt_5,879). happensAt(walking(lbbcgt_5),879).
orientation(lbbcgt_4,137,880). coords(lbbcgt_4,9,63,880).  visible(lbbcgt_4,880). happensAt(walking(lbbcgt_4),880). orientation(lbbcgt_5,64,880). coords(lbbcgt_5,250,281,880).  visible(lbbcgt_5,880). happensAt(walking(lbbcgt_5),880).
orientation(lbbcgt_4,137,881). coords(lbbcgt_4,9,62,881).  visible(lbbcgt_4,881). happensAt(walking(lbbcgt_4),881). orientation(lbbcgt_5,64,881). coords(lbbcgt_5,249,281,881).  visible(lbbcgt_5,881). happensAt(walking(lbbcgt_5),881).
orientation(lbbcgt_4,137,882). coords(lbbcgt_4,9,61,882).  visible(lbbcgt_4,882). happensAt(walking(lbbcgt_4),882). orientation(lbbcgt_5,64,882). coords(lbbcgt_5,249,281,882).  visible(lbbcgt_5,882). happensAt(walking(lbbcgt_5),882).
orientation(lbbcgt_4,132,883). coords(lbbcgt_4,9,61,883).  visible(lbbcgt_4,883). happensAt(walking(lbbcgt_4),883). orientation(lbbcgt_5,64,883). coords(lbbcgt_5,248,281,883).  visible(lbbcgt_5,883). happensAt(walking(lbbcgt_5),883).
orientation(lbbcgt_4,132,884). coords(lbbcgt_4,10,60,884).  visible(lbbcgt_4,884). happensAt(walking(lbbcgt_4),884). orientation(lbbcgt_5,20,884). coords(lbbcgt_5,243,281,884).  visible(lbbcgt_5,884). happensAt(walking(lbbcgt_5),884).
orientation(lbbcgt_4,132,885). coords(lbbcgt_4,13,59,885).  visible(lbbcgt_4,885). happensAt(walking(lbbcgt_4),885). orientation(lbbcgt_5,20,885). coords(lbbcgt_5,241,280,885).  visible(lbbcgt_5,885). happensAt(walking(lbbcgt_5),885).
orientation(lbbcgt_4,132,886). coords(lbbcgt_4,13,57,886).  visible(lbbcgt_4,886). happensAt(walking(lbbcgt_4),886). orientation(lbbcgt_5,20,886). coords(lbbcgt_5,239,278,886).  visible(lbbcgt_5,886). happensAt(walking(lbbcgt_5),886).
orientation(lbbcgt_4,137,887). coords(lbbcgt_4,13,56,887).  visible(lbbcgt_4,887). happensAt(walking(lbbcgt_4),887). orientation(lbbcgt_5,20,887). coords(lbbcgt_5,238,274,887).  visible(lbbcgt_5,887). happensAt(walking(lbbcgt_5),887).
orientation(lbbcgt_4,137,888). coords(lbbcgt_4,13,55,888).  visible(lbbcgt_4,888). happensAt(walking(lbbcgt_4),888). orientation(lbbcgt_5,20,888). coords(lbbcgt_5,237,271,888).  visible(lbbcgt_5,888). happensAt(walking(lbbcgt_5),888).
orientation(lbbcgt_4,137,889). coords(lbbcgt_4,13,54,889).  visible(lbbcgt_4,889). happensAt(walking(lbbcgt_4),889). orientation(lbbcgt_5,20,889). coords(lbbcgt_5,236,268,889).  visible(lbbcgt_5,889). happensAt(walking(lbbcgt_5),889).
orientation(lbbcgt_4,137,890). coords(lbbcgt_4,13,54,890).  visible(lbbcgt_4,890). happensAt(walking(lbbcgt_4),890). orientation(lbbcgt_5,20,890). coords(lbbcgt_5,236,267,890).  visible(lbbcgt_5,890). happensAt(walking(lbbcgt_5),890).
orientation(lbbcgt_4,137,891). coords(lbbcgt_4,13,53,891).  happensAt(occluded(lbbcgt_4),891). happensAt(walking(lbbcgt_4),891). orientation(lbbcgt_5,21,891). coords(lbbcgt_5,236,266,891).  visible(lbbcgt_5,891). happensAt(walking(lbbcgt_5),891).
orientation(lbbcgt_4,137,892). coords(lbbcgt_4,13,53,892).  happensAt(occluded(lbbcgt_4),892). happensAt(walking(lbbcgt_4),892). orientation(lbbcgt_5,21,892). coords(lbbcgt_5,236,266,892).  visible(lbbcgt_5,892). happensAt(walking(lbbcgt_5),892).
orientation(lbbcgt_4,137,893). coords(lbbcgt_4,13,53,893).  happensAt(occluded(lbbcgt_4),893). happensAt(walking(lbbcgt_4),893). orientation(lbbcgt_5,21,893). coords(lbbcgt_5,235,265,893).  visible(lbbcgt_5,893). happensAt(walking(lbbcgt_5),893).
orientation(lbbcgt_4,137,894). coords(lbbcgt_4,13,53,894).  happensAt(occluded(lbbcgt_4),894). happensAt(walking(lbbcgt_4),894). orientation(lbbcgt_5,21,894). coords(lbbcgt_5,234,264,894).  visible(lbbcgt_5,894). happensAt(walking(lbbcgt_5),894).
orientation(lbbcgt_4,137,895). coords(lbbcgt_4,17,46,895).  visible(lbbcgt_4,895). happensAt(walking(lbbcgt_4),895). orientation(lbbcgt_5,21,895). coords(lbbcgt_5,234,263,895).  visible(lbbcgt_5,895). happensAt(walking(lbbcgt_5),895).
orientation(lbbcgt_4,140,896). coords(lbbcgt_4,17,46,896).  visible(lbbcgt_4,896). happensAt(walking(lbbcgt_4),896). orientation(lbbcgt_5,21,896). coords(lbbcgt_5,231,261,896).  visible(lbbcgt_5,896). happensAt(walking(lbbcgt_5),896).
orientation(lbbcgt_4,144,897). coords(lbbcgt_4,17,45,897).  visible(lbbcgt_4,897). happensAt(walking(lbbcgt_4),897). orientation(lbbcgt_5,21,897). coords(lbbcgt_5,231,259,897).  visible(lbbcgt_5,897). happensAt(walking(lbbcgt_5),897).
#show holdsAt/2.