
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

	orientation(wk2gt_1,127,876). coords(wk2gt_1,115,53,876).  visible(wk2gt_1,876). happensAt(inactive(wk2gt_1),876). orientation(wk2gt_2,142,876). coords(wk2gt_2,99,63,876).  visible(wk2gt_2,876). happensAt(inactive(wk2gt_2),876).
orientation(wk2gt_1,127,877). coords(wk2gt_1,115,53,877).  visible(wk2gt_1,877). happensAt(inactive(wk2gt_1),877). orientation(wk2gt_2,142,877). coords(wk2gt_2,99,63,877).  visible(wk2gt_2,877). happensAt(inactive(wk2gt_2),877).
orientation(wk2gt_1,127,878). coords(wk2gt_1,115,53,878).  visible(wk2gt_1,878). happensAt(inactive(wk2gt_1),878). orientation(wk2gt_2,142,878). coords(wk2gt_2,99,63,878).  visible(wk2gt_2,878). happensAt(inactive(wk2gt_2),878).
orientation(wk2gt_1,127,879). coords(wk2gt_1,115,53,879).  visible(wk2gt_1,879). happensAt(inactive(wk2gt_1),879). orientation(wk2gt_2,142,879). coords(wk2gt_2,99,63,879).  visible(wk2gt_2,879). happensAt(inactive(wk2gt_2),879).
orientation(wk2gt_1,127,880). coords(wk2gt_1,115,53,880).  visible(wk2gt_1,880). happensAt(inactive(wk2gt_1),880). orientation(wk2gt_2,142,880). coords(wk2gt_2,99,63,880).  visible(wk2gt_2,880). happensAt(inactive(wk2gt_2),880).
orientation(wk2gt_1,127,881). coords(wk2gt_1,115,53,881).  visible(wk2gt_1,881). happensAt(inactive(wk2gt_1),881). orientation(wk2gt_2,142,881). coords(wk2gt_2,99,63,881).  visible(wk2gt_2,881). happensAt(inactive(wk2gt_2),881).
orientation(wk2gt_1,127,882). coords(wk2gt_1,115,53,882).  visible(wk2gt_1,882). happensAt(inactive(wk2gt_1),882). orientation(wk2gt_2,142,882). coords(wk2gt_2,99,63,882).  visible(wk2gt_2,882). happensAt(inactive(wk2gt_2),882).
orientation(wk2gt_1,127,883). coords(wk2gt_1,115,53,883).  visible(wk2gt_1,883). happensAt(inactive(wk2gt_1),883). orientation(wk2gt_2,142,883). coords(wk2gt_2,99,63,883).  visible(wk2gt_2,883). happensAt(inactive(wk2gt_2),883).
orientation(wk2gt_1,127,884). coords(wk2gt_1,115,53,884).  visible(wk2gt_1,884). happensAt(inactive(wk2gt_1),884). orientation(wk2gt_2,142,884). coords(wk2gt_2,99,63,884).  visible(wk2gt_2,884). happensAt(inactive(wk2gt_2),884).
orientation(wk2gt_1,127,885). coords(wk2gt_1,115,53,885).  visible(wk2gt_1,885). happensAt(inactive(wk2gt_1),885). orientation(wk2gt_2,142,885). coords(wk2gt_2,99,63,885).  visible(wk2gt_2,885). happensAt(inactive(wk2gt_2),885).
orientation(wk2gt_1,127,886). coords(wk2gt_1,115,53,886).  visible(wk2gt_1,886). happensAt(inactive(wk2gt_1),886). orientation(wk2gt_2,142,886). coords(wk2gt_2,99,63,886).  visible(wk2gt_2,886). happensAt(inactive(wk2gt_2),886).
orientation(wk2gt_1,127,887). coords(wk2gt_1,115,53,887).  visible(wk2gt_1,887). happensAt(inactive(wk2gt_1),887). orientation(wk2gt_2,142,887). coords(wk2gt_2,99,63,887).  visible(wk2gt_2,887). happensAt(inactive(wk2gt_2),887).
orientation(wk2gt_1,127,888). coords(wk2gt_1,115,53,888).  visible(wk2gt_1,888). happensAt(inactive(wk2gt_1),888). orientation(wk2gt_2,142,888). coords(wk2gt_2,99,63,888).  visible(wk2gt_2,888). happensAt(inactive(wk2gt_2),888).
orientation(wk2gt_1,127,889). coords(wk2gt_1,115,53,889).  visible(wk2gt_1,889). happensAt(inactive(wk2gt_1),889). orientation(wk2gt_2,142,889). coords(wk2gt_2,99,63,889).  visible(wk2gt_2,889). happensAt(inactive(wk2gt_2),889).
orientation(wk2gt_1,127,890). coords(wk2gt_1,115,53,890).  visible(wk2gt_1,890). happensAt(inactive(wk2gt_1),890). orientation(wk2gt_2,142,890). coords(wk2gt_2,99,63,890).  visible(wk2gt_2,890). happensAt(inactive(wk2gt_2),890).
orientation(wk2gt_1,127,891). coords(wk2gt_1,115,53,891).  visible(wk2gt_1,891). happensAt(inactive(wk2gt_1),891). orientation(wk2gt_2,142,891). coords(wk2gt_2,99,63,891).  visible(wk2gt_2,891). happensAt(inactive(wk2gt_2),891).
orientation(wk2gt_1,127,892). coords(wk2gt_1,115,53,892).  visible(wk2gt_1,892). happensAt(inactive(wk2gt_1),892). orientation(wk2gt_2,142,892). coords(wk2gt_2,99,63,892).  visible(wk2gt_2,892). happensAt(inactive(wk2gt_2),892).
orientation(wk2gt_1,127,893). coords(wk2gt_1,115,53,893).  visible(wk2gt_1,893). happensAt(inactive(wk2gt_1),893). orientation(wk2gt_2,142,893). coords(wk2gt_2,99,63,893).  visible(wk2gt_2,893). happensAt(inactive(wk2gt_2),893).
orientation(wk2gt_1,127,894). coords(wk2gt_1,115,53,894).  visible(wk2gt_1,894). happensAt(inactive(wk2gt_1),894). orientation(wk2gt_2,142,894). coords(wk2gt_2,99,63,894).  visible(wk2gt_2,894). happensAt(inactive(wk2gt_2),894).
orientation(wk2gt_1,127,895). coords(wk2gt_1,115,53,895).  visible(wk2gt_1,895). happensAt(inactive(wk2gt_1),895). orientation(wk2gt_2,142,895). coords(wk2gt_2,99,63,895).  visible(wk2gt_2,895). happensAt(inactive(wk2gt_2),895).
orientation(wk2gt_1,127,896). coords(wk2gt_1,115,53,896).  visible(wk2gt_1,896). happensAt(inactive(wk2gt_1),896). orientation(wk2gt_2,142,896). coords(wk2gt_2,99,63,896).  visible(wk2gt_2,896). happensAt(inactive(wk2gt_2),896).
orientation(wk2gt_1,127,897). coords(wk2gt_1,115,53,897).  visible(wk2gt_1,897). happensAt(inactive(wk2gt_1),897). orientation(wk2gt_2,142,897). coords(wk2gt_2,99,63,897).  visible(wk2gt_2,897). happensAt(inactive(wk2gt_2),897).
orientation(wk2gt_1,127,898). coords(wk2gt_1,115,53,898).  visible(wk2gt_1,898). happensAt(inactive(wk2gt_1),898). orientation(wk2gt_2,142,898). coords(wk2gt_2,99,63,898).  visible(wk2gt_2,898). happensAt(inactive(wk2gt_2),898).
orientation(wk2gt_1,127,899). coords(wk2gt_1,115,53,899).  visible(wk2gt_1,899). happensAt(inactive(wk2gt_1),899). orientation(wk2gt_2,142,899). coords(wk2gt_2,99,63,899).  visible(wk2gt_2,899). happensAt(inactive(wk2gt_2),899).
orientation(wk2gt_1,127,900). coords(wk2gt_1,115,53,900).  visible(wk2gt_1,900). happensAt(inactive(wk2gt_1),900). orientation(wk2gt_2,142,900). coords(wk2gt_2,99,63,900).  visible(wk2gt_2,900). happensAt(inactive(wk2gt_2),900).
#show holdsAt/2.