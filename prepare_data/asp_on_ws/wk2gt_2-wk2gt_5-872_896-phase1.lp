
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

	orientation(wk2gt_2,142,872). coords(wk2gt_2,99,63,872).  visible(wk2gt_2,872). happensAt(inactive(wk2gt_2),872). orientation(wk2gt_5,156,872). coords(wk2gt_5,48,100,872).  visible(wk2gt_5,872). happensAt(walking(wk2gt_5),872).
orientation(wk2gt_2,142,873). coords(wk2gt_2,99,63,873).  visible(wk2gt_2,873). happensAt(inactive(wk2gt_2),873). orientation(wk2gt_5,156,873). coords(wk2gt_5,48,100,873).  visible(wk2gt_5,873). happensAt(walking(wk2gt_5),873).
orientation(wk2gt_2,142,874). coords(wk2gt_2,99,63,874).  visible(wk2gt_2,874). happensAt(inactive(wk2gt_2),874). orientation(wk2gt_5,156,874). coords(wk2gt_5,48,100,874).  visible(wk2gt_5,874). happensAt(walking(wk2gt_5),874).
orientation(wk2gt_2,142,875). coords(wk2gt_2,99,63,875).  visible(wk2gt_2,875). happensAt(inactive(wk2gt_2),875). orientation(wk2gt_5,156,875). coords(wk2gt_5,49,100,875).  visible(wk2gt_5,875). happensAt(walking(wk2gt_5),875).
orientation(wk2gt_2,142,876). coords(wk2gt_2,99,63,876).  visible(wk2gt_2,876). happensAt(inactive(wk2gt_2),876). orientation(wk2gt_5,156,876). coords(wk2gt_5,49,100,876).  visible(wk2gt_5,876). happensAt(walking(wk2gt_5),876).
orientation(wk2gt_2,142,877). coords(wk2gt_2,99,63,877).  visible(wk2gt_2,877). happensAt(inactive(wk2gt_2),877). orientation(wk2gt_5,156,877). coords(wk2gt_5,49,100,877).  visible(wk2gt_5,877). happensAt(walking(wk2gt_5),877).
orientation(wk2gt_2,142,878). coords(wk2gt_2,99,63,878).  visible(wk2gt_2,878). happensAt(inactive(wk2gt_2),878). orientation(wk2gt_5,156,878). coords(wk2gt_5,50,100,878).  visible(wk2gt_5,878). happensAt(walking(wk2gt_5),878).
orientation(wk2gt_2,142,879). coords(wk2gt_2,99,63,879).  visible(wk2gt_2,879). happensAt(inactive(wk2gt_2),879). orientation(wk2gt_5,156,879). coords(wk2gt_5,52,100,879).  visible(wk2gt_5,879). happensAt(walking(wk2gt_5),879).
orientation(wk2gt_2,142,880). coords(wk2gt_2,99,63,880).  visible(wk2gt_2,880). happensAt(inactive(wk2gt_2),880). orientation(wk2gt_5,156,880). coords(wk2gt_5,52,100,880).  visible(wk2gt_5,880). happensAt(walking(wk2gt_5),880).
orientation(wk2gt_2,142,881). coords(wk2gt_2,99,63,881).  visible(wk2gt_2,881). happensAt(inactive(wk2gt_2),881). orientation(wk2gt_5,156,881). coords(wk2gt_5,52,100,881).  visible(wk2gt_5,881). happensAt(walking(wk2gt_5),881).
orientation(wk2gt_2,142,882). coords(wk2gt_2,99,63,882).  visible(wk2gt_2,882). happensAt(inactive(wk2gt_2),882). orientation(wk2gt_5,156,882). coords(wk2gt_5,52,99,882).  visible(wk2gt_5,882). happensAt(walking(wk2gt_5),882).
orientation(wk2gt_2,142,883). coords(wk2gt_2,99,63,883).  visible(wk2gt_2,883). happensAt(inactive(wk2gt_2),883). orientation(wk2gt_5,156,883). coords(wk2gt_5,52,99,883).  visible(wk2gt_5,883). happensAt(walking(wk2gt_5),883).
orientation(wk2gt_2,142,884). coords(wk2gt_2,99,63,884).  visible(wk2gt_2,884). happensAt(inactive(wk2gt_2),884). orientation(wk2gt_5,156,884). coords(wk2gt_5,53,99,884).  visible(wk2gt_5,884). happensAt(walking(wk2gt_5),884).
orientation(wk2gt_2,142,885). coords(wk2gt_2,99,63,885).  visible(wk2gt_2,885). happensAt(inactive(wk2gt_2),885). orientation(wk2gt_5,156,885). coords(wk2gt_5,53,99,885).  visible(wk2gt_5,885). happensAt(walking(wk2gt_5),885).
orientation(wk2gt_2,142,886). coords(wk2gt_2,99,63,886).  visible(wk2gt_2,886). happensAt(inactive(wk2gt_2),886). orientation(wk2gt_5,156,886). coords(wk2gt_5,53,99,886).  visible(wk2gt_5,886). happensAt(walking(wk2gt_5),886).
orientation(wk2gt_2,142,887). coords(wk2gt_2,99,63,887).  visible(wk2gt_2,887). happensAt(inactive(wk2gt_2),887). orientation(wk2gt_5,156,887). coords(wk2gt_5,53,99,887).  visible(wk2gt_5,887). happensAt(walking(wk2gt_5),887).
orientation(wk2gt_2,142,888). coords(wk2gt_2,99,63,888).  visible(wk2gt_2,888). happensAt(inactive(wk2gt_2),888). orientation(wk2gt_5,156,888). coords(wk2gt_5,53,99,888).  visible(wk2gt_5,888). happensAt(walking(wk2gt_5),888).
orientation(wk2gt_2,142,889). coords(wk2gt_2,99,63,889).  visible(wk2gt_2,889). happensAt(inactive(wk2gt_2),889). orientation(wk2gt_5,156,889). coords(wk2gt_5,52,97,889).  visible(wk2gt_5,889). happensAt(walking(wk2gt_5),889).
orientation(wk2gt_2,142,890). coords(wk2gt_2,99,63,890).  visible(wk2gt_2,890). happensAt(inactive(wk2gt_2),890). orientation(wk2gt_5,156,890). coords(wk2gt_5,52,97,890).  visible(wk2gt_5,890). happensAt(walking(wk2gt_5),890).
orientation(wk2gt_2,142,891). coords(wk2gt_2,99,63,891).  visible(wk2gt_2,891). happensAt(inactive(wk2gt_2),891). orientation(wk2gt_5,156,891). coords(wk2gt_5,55,98,891).  visible(wk2gt_5,891). happensAt(walking(wk2gt_5),891).
orientation(wk2gt_2,142,892). coords(wk2gt_2,99,63,892).  visible(wk2gt_2,892). happensAt(inactive(wk2gt_2),892). orientation(wk2gt_5,156,892). coords(wk2gt_5,56,98,892).  visible(wk2gt_5,892). happensAt(walking(wk2gt_5),892).
orientation(wk2gt_2,142,893). coords(wk2gt_2,99,63,893).  visible(wk2gt_2,893). happensAt(inactive(wk2gt_2),893). orientation(wk2gt_5,156,893). coords(wk2gt_5,56,97,893).  visible(wk2gt_5,893). happensAt(walking(wk2gt_5),893).
orientation(wk2gt_2,142,894). coords(wk2gt_2,99,63,894).  visible(wk2gt_2,894). happensAt(inactive(wk2gt_2),894). orientation(wk2gt_5,156,894). coords(wk2gt_5,57,97,894).  visible(wk2gt_5,894). happensAt(walking(wk2gt_5),894).
orientation(wk2gt_2,142,895). coords(wk2gt_2,99,63,895).  visible(wk2gt_2,895). happensAt(inactive(wk2gt_2),895). orientation(wk2gt_5,156,895). coords(wk2gt_5,57,97,895).  visible(wk2gt_5,895). happensAt(walking(wk2gt_5),895).
orientation(wk2gt_2,142,896). coords(wk2gt_2,99,63,896).  visible(wk2gt_2,896). happensAt(inactive(wk2gt_2),896). orientation(wk2gt_5,156,896). coords(wk2gt_5,57,97,896).  visible(wk2gt_5,896). happensAt(walking(wk2gt_5),896).
#show holdsAt/2.