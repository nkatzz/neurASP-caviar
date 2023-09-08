
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

	orientation(lbbcgt_5,103,873). coords(lbbcgt_5,252,285,873).  happensAt(appear(lbbcgt_5),873). happensAt(walking(lbbcgt_5),873). orientation(lbbcgt_6,130,873). coords(lbbcgt_6,156,43,873).  visible(lbbcgt_6,873). happensAt(running(lbbcgt_6),873).
orientation(lbbcgt_5,100,874). coords(lbbcgt_5,253,284,874).  visible(lbbcgt_5,874). happensAt(walking(lbbcgt_5),874). orientation(lbbcgt_6,130,874). coords(lbbcgt_6,154,44,874).  visible(lbbcgt_6,874). happensAt(running(lbbcgt_6),874).
orientation(lbbcgt_5,103,875). coords(lbbcgt_5,253,282,875).  visible(lbbcgt_5,875). happensAt(walking(lbbcgt_5),875). orientation(lbbcgt_6,130,875). coords(lbbcgt_6,153,44,875).  visible(lbbcgt_6,875). happensAt(running(lbbcgt_6),875).
orientation(lbbcgt_5,103,876). coords(lbbcgt_5,253,282,876).  visible(lbbcgt_5,876). happensAt(walking(lbbcgt_5),876). orientation(lbbcgt_6,130,876). coords(lbbcgt_6,152,45,876).  visible(lbbcgt_6,876). happensAt(running(lbbcgt_6),876).
orientation(lbbcgt_5,103,877). coords(lbbcgt_5,252,281,877).  visible(lbbcgt_5,877). happensAt(walking(lbbcgt_5),877). orientation(lbbcgt_6,133,877). coords(lbbcgt_6,151,45,877).  visible(lbbcgt_6,877). happensAt(running(lbbcgt_6),877).
orientation(lbbcgt_5,103,878). coords(lbbcgt_5,252,281,878).  visible(lbbcgt_5,878). happensAt(walking(lbbcgt_5),878). orientation(lbbcgt_6,138,878). coords(lbbcgt_6,149,45,878).  visible(lbbcgt_6,878). happensAt(running(lbbcgt_6),878).
orientation(lbbcgt_5,64,879). coords(lbbcgt_5,251,281,879).  visible(lbbcgt_5,879). happensAt(walking(lbbcgt_5),879). orientation(lbbcgt_6,135,879). coords(lbbcgt_6,149,47,879).  visible(lbbcgt_6,879). happensAt(running(lbbcgt_6),879).
orientation(lbbcgt_5,64,880). coords(lbbcgt_5,250,281,880).  visible(lbbcgt_5,880). happensAt(walking(lbbcgt_5),880). orientation(lbbcgt_6,134,880). coords(lbbcgt_6,148,50,880).  visible(lbbcgt_6,880). happensAt(running(lbbcgt_6),880).
orientation(lbbcgt_5,64,881). coords(lbbcgt_5,249,281,881).  visible(lbbcgt_5,881). happensAt(walking(lbbcgt_5),881). orientation(lbbcgt_6,135,881). coords(lbbcgt_6,147,51,881).  visible(lbbcgt_6,881). happensAt(running(lbbcgt_6),881).
orientation(lbbcgt_5,64,882). coords(lbbcgt_5,249,281,882).  visible(lbbcgt_5,882). happensAt(walking(lbbcgt_5),882). orientation(lbbcgt_6,142,882). coords(lbbcgt_6,148,53,882).  visible(lbbcgt_6,882). happensAt(running(lbbcgt_6),882).
orientation(lbbcgt_5,64,883). coords(lbbcgt_5,248,281,883).  visible(lbbcgt_5,883). happensAt(walking(lbbcgt_5),883). orientation(lbbcgt_6,142,883). coords(lbbcgt_6,147,54,883).  visible(lbbcgt_6,883). happensAt(running(lbbcgt_6),883).
orientation(lbbcgt_5,20,884). coords(lbbcgt_5,243,281,884).  visible(lbbcgt_5,884). happensAt(walking(lbbcgt_5),884). orientation(lbbcgt_6,142,884). coords(lbbcgt_6,146,55,884).  visible(lbbcgt_6,884). happensAt(walking(lbbcgt_6),884).
orientation(lbbcgt_5,20,885). coords(lbbcgt_5,241,280,885).  visible(lbbcgt_5,885). happensAt(walking(lbbcgt_5),885). orientation(lbbcgt_6,142,885). coords(lbbcgt_6,145,56,885).  visible(lbbcgt_6,885). happensAt(walking(lbbcgt_6),885).
orientation(lbbcgt_5,20,886). coords(lbbcgt_5,239,278,886).  visible(lbbcgt_5,886). happensAt(walking(lbbcgt_5),886). orientation(lbbcgt_6,142,886). coords(lbbcgt_6,143,57,886).  visible(lbbcgt_6,886). happensAt(walking(lbbcgt_6),886).
orientation(lbbcgt_5,20,887). coords(lbbcgt_5,238,274,887).  visible(lbbcgt_5,887). happensAt(walking(lbbcgt_5),887). orientation(lbbcgt_6,142,887). coords(lbbcgt_6,142,58,887).  visible(lbbcgt_6,887). happensAt(walking(lbbcgt_6),887).
orientation(lbbcgt_5,20,888). coords(lbbcgt_5,237,271,888).  visible(lbbcgt_5,888). happensAt(walking(lbbcgt_5),888). orientation(lbbcgt_6,142,888). coords(lbbcgt_6,140,58,888).  visible(lbbcgt_6,888). happensAt(walking(lbbcgt_6),888).
orientation(lbbcgt_5,20,889). coords(lbbcgt_5,236,268,889).  visible(lbbcgt_5,889). happensAt(walking(lbbcgt_5),889). orientation(lbbcgt_6,142,889). coords(lbbcgt_6,138,58,889).  visible(lbbcgt_6,889). happensAt(walking(lbbcgt_6),889).
orientation(lbbcgt_5,20,890). coords(lbbcgt_5,236,267,890).  visible(lbbcgt_5,890). happensAt(walking(lbbcgt_5),890). orientation(lbbcgt_6,142,890). coords(lbbcgt_6,136,59,890).  visible(lbbcgt_6,890). happensAt(walking(lbbcgt_6),890).
orientation(lbbcgt_5,21,891). coords(lbbcgt_5,236,266,891).  visible(lbbcgt_5,891). happensAt(walking(lbbcgt_5),891). orientation(lbbcgt_6,142,891). coords(lbbcgt_6,135,60,891).  visible(lbbcgt_6,891). happensAt(walking(lbbcgt_6),891).
orientation(lbbcgt_5,21,892). coords(lbbcgt_5,236,266,892).  visible(lbbcgt_5,892). happensAt(walking(lbbcgt_5),892). orientation(lbbcgt_6,142,892). coords(lbbcgt_6,134,63,892).  visible(lbbcgt_6,892). happensAt(walking(lbbcgt_6),892).
orientation(lbbcgt_5,21,893). coords(lbbcgt_5,235,265,893).  visible(lbbcgt_5,893). happensAt(walking(lbbcgt_5),893). orientation(lbbcgt_6,142,893). coords(lbbcgt_6,134,64,893).  visible(lbbcgt_6,893). happensAt(walking(lbbcgt_6),893).
orientation(lbbcgt_5,21,894). coords(lbbcgt_5,234,264,894).  visible(lbbcgt_5,894). happensAt(walking(lbbcgt_5),894). orientation(lbbcgt_6,142,894). coords(lbbcgt_6,133,66,894).  visible(lbbcgt_6,894). happensAt(walking(lbbcgt_6),894).
orientation(lbbcgt_5,21,895). coords(lbbcgt_5,234,263,895).  visible(lbbcgt_5,895). happensAt(walking(lbbcgt_5),895). orientation(lbbcgt_6,142,895). coords(lbbcgt_6,133,67,895).  visible(lbbcgt_6,895). happensAt(walking(lbbcgt_6),895).
orientation(lbbcgt_5,21,896). coords(lbbcgt_5,231,261,896).  visible(lbbcgt_5,896). happensAt(walking(lbbcgt_5),896). orientation(lbbcgt_6,142,896). coords(lbbcgt_6,132,68,896).  visible(lbbcgt_6,896). happensAt(walking(lbbcgt_6),896).
orientation(lbbcgt_5,21,897). coords(lbbcgt_5,231,259,897).  visible(lbbcgt_5,897). happensAt(walking(lbbcgt_5),897). orientation(lbbcgt_6,142,897). coords(lbbcgt_6,132,68,897).  visible(lbbcgt_6,897). happensAt(walking(lbbcgt_6),897).
#show holdsAt/2.