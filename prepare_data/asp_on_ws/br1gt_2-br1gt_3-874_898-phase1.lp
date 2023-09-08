
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

	orientation(br1gt_2,5,874). coords(br1gt_2,85,206,874).  visible(br1gt_2,874). happensAt(inactive(br1gt_2),874). orientation(br1gt_3,104,874). coords(br1gt_3,293,170,874).  visible(br1gt_3,874). happensAt(walking(br1gt_3),874).
orientation(br1gt_2,5,875). coords(br1gt_2,85,205,875).  visible(br1gt_2,875). happensAt(inactive(br1gt_2),875). orientation(br1gt_3,104,875). coords(br1gt_3,293,169,875).  visible(br1gt_3,875). happensAt(walking(br1gt_3),875).
orientation(br1gt_2,5,876). coords(br1gt_2,85,205,876).  visible(br1gt_2,876). happensAt(inactive(br1gt_2),876). orientation(br1gt_3,104,876). coords(br1gt_3,293,168,876).  visible(br1gt_3,876). happensAt(walking(br1gt_3),876).
orientation(br1gt_2,5,877). coords(br1gt_2,85,205,877).  visible(br1gt_2,877). happensAt(inactive(br1gt_2),877). orientation(br1gt_3,104,877). coords(br1gt_3,294,167,877).  visible(br1gt_3,877). happensAt(walking(br1gt_3),877).
orientation(br1gt_2,5,878). coords(br1gt_2,85,205,878).  visible(br1gt_2,878). happensAt(inactive(br1gt_2),878). orientation(br1gt_3,104,878). coords(br1gt_3,294,166,878).  visible(br1gt_3,878). happensAt(walking(br1gt_3),878).
orientation(br1gt_2,5,879). coords(br1gt_2,85,205,879).  visible(br1gt_2,879). happensAt(inactive(br1gt_2),879). orientation(br1gt_3,104,879). coords(br1gt_3,293,164,879).  visible(br1gt_3,879). happensAt(walking(br1gt_3),879).
orientation(br1gt_2,5,880). coords(br1gt_2,86,205,880).  visible(br1gt_2,880). happensAt(inactive(br1gt_2),880). orientation(br1gt_3,104,880). coords(br1gt_3,293,163,880).  visible(br1gt_3,880). happensAt(walking(br1gt_3),880).
orientation(br1gt_2,5,881). coords(br1gt_2,85,204,881).  visible(br1gt_2,881). happensAt(inactive(br1gt_2),881). orientation(br1gt_3,104,881). coords(br1gt_3,292,162,881).  visible(br1gt_3,881). happensAt(walking(br1gt_3),881).
orientation(br1gt_2,5,882). coords(br1gt_2,85,204,882).  visible(br1gt_2,882). happensAt(inactive(br1gt_2),882). orientation(br1gt_3,104,882). coords(br1gt_3,289,160,882).  visible(br1gt_3,882). happensAt(walking(br1gt_3),882).
orientation(br1gt_2,5,883). coords(br1gt_2,85,204,883).  visible(br1gt_2,883). happensAt(inactive(br1gt_2),883). orientation(br1gt_3,104,883). coords(br1gt_3,289,157,883).  visible(br1gt_3,883). happensAt(walking(br1gt_3),883).
orientation(br1gt_2,2,884). coords(br1gt_2,85,204,884).  visible(br1gt_2,884). happensAt(inactive(br1gt_2),884). orientation(br1gt_3,104,884). coords(br1gt_3,289,154,884).  visible(br1gt_3,884). happensAt(walking(br1gt_3),884).
orientation(br1gt_2,2,885). coords(br1gt_2,85,204,885).  visible(br1gt_2,885). happensAt(inactive(br1gt_2),885). orientation(br1gt_3,104,885). coords(br1gt_3,289,152,885).  visible(br1gt_3,885). happensAt(walking(br1gt_3),885).
orientation(br1gt_2,2,886). coords(br1gt_2,85,204,886).  visible(br1gt_2,886). happensAt(inactive(br1gt_2),886). orientation(br1gt_3,104,886). coords(br1gt_3,288,149,886).  visible(br1gt_3,886). happensAt(walking(br1gt_3),886).
orientation(br1gt_2,2,887). coords(br1gt_2,85,204,887).  visible(br1gt_2,887). happensAt(inactive(br1gt_2),887). orientation(br1gt_3,110,887). coords(br1gt_3,287,146,887).  visible(br1gt_3,887). happensAt(walking(br1gt_3),887).
orientation(br1gt_2,2,888). coords(br1gt_2,85,203,888).  visible(br1gt_2,888). happensAt(inactive(br1gt_2),888). orientation(br1gt_3,110,888). coords(br1gt_3,287,143,888).  visible(br1gt_3,888). happensAt(walking(br1gt_3),888).
orientation(br1gt_2,2,889). coords(br1gt_2,85,203,889).  visible(br1gt_2,889). happensAt(active(br1gt_2),889). orientation(br1gt_3,110,889). coords(br1gt_3,287,139,889).  visible(br1gt_3,889). happensAt(walking(br1gt_3),889).
orientation(br1gt_2,2,890). coords(br1gt_2,85,203,890).  visible(br1gt_2,890). happensAt(active(br1gt_2),890). orientation(br1gt_3,108,890). coords(br1gt_3,288,138,890).  visible(br1gt_3,890). happensAt(walking(br1gt_3),890).
orientation(br1gt_2,2,891). coords(br1gt_2,85,202,891).  visible(br1gt_2,891). happensAt(active(br1gt_2),891). orientation(br1gt_3,106,891). coords(br1gt_3,288,137,891).  visible(br1gt_3,891). happensAt(walking(br1gt_3),891).
orientation(br1gt_2,2,892). coords(br1gt_2,85,203,892).  visible(br1gt_2,892). happensAt(active(br1gt_2),892). orientation(br1gt_3,106,892). coords(br1gt_3,288,136,892).  visible(br1gt_3,892). happensAt(walking(br1gt_3),892).
orientation(br1gt_2,2,893). coords(br1gt_2,85,203,893).  visible(br1gt_2,893). happensAt(active(br1gt_2),893). orientation(br1gt_3,101,893). coords(br1gt_3,287,136,893).  visible(br1gt_3,893). happensAt(walking(br1gt_3),893).
orientation(br1gt_2,2,894). coords(br1gt_2,85,203,894).  visible(br1gt_2,894). happensAt(active(br1gt_2),894). orientation(br1gt_3,97,894). coords(br1gt_3,287,135,894).  visible(br1gt_3,894). happensAt(walking(br1gt_3),894).
orientation(br1gt_2,2,895). coords(br1gt_2,85,202,895).  visible(br1gt_2,895). happensAt(active(br1gt_2),895). orientation(br1gt_3,97,895). coords(br1gt_3,288,134,895).  visible(br1gt_3,895). happensAt(walking(br1gt_3),895).
orientation(br1gt_2,2,896). coords(br1gt_2,85,203,896).  visible(br1gt_2,896). happensAt(active(br1gt_2),896). orientation(br1gt_3,94,896). coords(br1gt_3,288,134,896).  visible(br1gt_3,896). happensAt(walking(br1gt_3),896).
orientation(br1gt_2,2,897). coords(br1gt_2,85,202,897).  visible(br1gt_2,897). happensAt(active(br1gt_2),897). orientation(br1gt_3,94,897). coords(br1gt_3,288,133,897).  visible(br1gt_3,897). happensAt(walking(br1gt_3),897).
orientation(br1gt_2,2,898). coords(br1gt_2,85,201,898).  visible(br1gt_2,898). happensAt(active(br1gt_2),898). orientation(br1gt_3,94,898). coords(br1gt_3,288,131,898).  visible(br1gt_3,898). happensAt(walking(br1gt_3),898).
#show holdsAt/2.