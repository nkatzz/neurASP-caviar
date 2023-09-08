
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

	orientation(lb2gt_5,152,879). coords(lb2gt_5,73,111,879).  visible(lb2gt_5,879). happensAt(walking(lb2gt_5),879). orientation(lb2gt_6,151,879). coords(lb2gt_6,72,122,879).  visible(lb2gt_6,879). happensAt(inactive(lb2gt_6),879).
orientation(lb2gt_5,152,880). coords(lb2gt_5,72,111,880).  visible(lb2gt_5,880). happensAt(walking(lb2gt_5),880). orientation(lb2gt_6,151,880). coords(lb2gt_6,72,122,880).  visible(lb2gt_6,880). happensAt(inactive(lb2gt_6),880).
orientation(lb2gt_5,152,881). coords(lb2gt_5,72,111,881).  visible(lb2gt_5,881). happensAt(walking(lb2gt_5),881). orientation(lb2gt_6,151,881). coords(lb2gt_6,72,122,881).  visible(lb2gt_6,881). happensAt(inactive(lb2gt_6),881).
orientation(lb2gt_5,152,882). coords(lb2gt_5,72,112,882).  visible(lb2gt_5,882). happensAt(walking(lb2gt_5),882). orientation(lb2gt_6,151,882). coords(lb2gt_6,72,122,882).  visible(lb2gt_6,882). happensAt(inactive(lb2gt_6),882).
orientation(lb2gt_5,152,883). coords(lb2gt_5,72,112,883).  visible(lb2gt_5,883). happensAt(walking(lb2gt_5),883). orientation(lb2gt_6,151,883). coords(lb2gt_6,72,122,883).  visible(lb2gt_6,883). happensAt(inactive(lb2gt_6),883).
orientation(lb2gt_5,152,884). coords(lb2gt_5,72,112,884).  visible(lb2gt_5,884). happensAt(walking(lb2gt_5),884). orientation(lb2gt_6,151,884). coords(lb2gt_6,72,122,884).  visible(lb2gt_6,884). happensAt(inactive(lb2gt_6),884).
orientation(lb2gt_5,152,885). coords(lb2gt_5,72,112,885).  visible(lb2gt_5,885). happensAt(walking(lb2gt_5),885). orientation(lb2gt_6,151,885). coords(lb2gt_6,72,122,885).  visible(lb2gt_6,885). happensAt(inactive(lb2gt_6),885).
orientation(lb2gt_5,152,886). coords(lb2gt_5,72,112,886).  visible(lb2gt_5,886). happensAt(walking(lb2gt_5),886). orientation(lb2gt_6,151,886). coords(lb2gt_6,72,122,886).  visible(lb2gt_6,886). happensAt(inactive(lb2gt_6),886).
orientation(lb2gt_5,152,887). coords(lb2gt_5,72,112,887).  visible(lb2gt_5,887). happensAt(walking(lb2gt_5),887). orientation(lb2gt_6,151,887). coords(lb2gt_6,72,122,887).  visible(lb2gt_6,887). happensAt(inactive(lb2gt_6),887).
orientation(lb2gt_5,152,888). coords(lb2gt_5,72,112,888).  visible(lb2gt_5,888). happensAt(walking(lb2gt_5),888). orientation(lb2gt_6,151,888). coords(lb2gt_6,72,122,888).  visible(lb2gt_6,888). happensAt(inactive(lb2gt_6),888).
orientation(lb2gt_5,152,889). coords(lb2gt_5,72,112,889).  visible(lb2gt_5,889). happensAt(walking(lb2gt_5),889). orientation(lb2gt_6,151,889). coords(lb2gt_6,72,122,889).  visible(lb2gt_6,889). happensAt(inactive(lb2gt_6),889).
orientation(lb2gt_5,152,890). coords(lb2gt_5,72,112,890).  visible(lb2gt_5,890). happensAt(walking(lb2gt_5),890). orientation(lb2gt_6,151,890). coords(lb2gt_6,72,122,890).  visible(lb2gt_6,890). happensAt(inactive(lb2gt_6),890).
orientation(lb2gt_5,152,891). coords(lb2gt_5,72,112,891).  visible(lb2gt_5,891). happensAt(walking(lb2gt_5),891). orientation(lb2gt_6,151,891). coords(lb2gt_6,72,122,891).  visible(lb2gt_6,891). happensAt(inactive(lb2gt_6),891).
orientation(lb2gt_5,152,892). coords(lb2gt_5,72,112,892).  visible(lb2gt_5,892). happensAt(walking(lb2gt_5),892). orientation(lb2gt_6,151,892). coords(lb2gt_6,72,122,892).  visible(lb2gt_6,892). happensAt(inactive(lb2gt_6),892).
orientation(lb2gt_5,152,893). coords(lb2gt_5,71,112,893).  visible(lb2gt_5,893). happensAt(walking(lb2gt_5),893). orientation(lb2gt_6,151,893). coords(lb2gt_6,72,122,893).  visible(lb2gt_6,893). happensAt(inactive(lb2gt_6),893).
orientation(lb2gt_5,152,894). coords(lb2gt_5,71,112,894).  visible(lb2gt_5,894). happensAt(walking(lb2gt_5),894). orientation(lb2gt_6,151,894). coords(lb2gt_6,72,122,894).  visible(lb2gt_6,894). happensAt(inactive(lb2gt_6),894).
orientation(lb2gt_5,152,895). coords(lb2gt_5,71,113,895).  visible(lb2gt_5,895). happensAt(walking(lb2gt_5),895). orientation(lb2gt_6,151,895). coords(lb2gt_6,72,122,895).  visible(lb2gt_6,895). happensAt(inactive(lb2gt_6),895).
orientation(lb2gt_5,152,896). coords(lb2gt_5,71,113,896).  visible(lb2gt_5,896). happensAt(walking(lb2gt_5),896). orientation(lb2gt_6,151,896). coords(lb2gt_6,72,122,896).  visible(lb2gt_6,896). happensAt(inactive(lb2gt_6),896).
orientation(lb2gt_5,152,897). coords(lb2gt_5,71,112,897).  visible(lb2gt_5,897). happensAt(walking(lb2gt_5),897). orientation(lb2gt_6,151,897). coords(lb2gt_6,72,122,897).  visible(lb2gt_6,897). happensAt(inactive(lb2gt_6),897).
orientation(lb2gt_5,152,898). coords(lb2gt_5,71,113,898).  visible(lb2gt_5,898). happensAt(walking(lb2gt_5),898). orientation(lb2gt_6,151,898). coords(lb2gt_6,72,122,898).  visible(lb2gt_6,898). happensAt(inactive(lb2gt_6),898).
orientation(lb2gt_5,152,899). coords(lb2gt_5,71,113,899).  visible(lb2gt_5,899). happensAt(walking(lb2gt_5),899). orientation(lb2gt_6,151,899). coords(lb2gt_6,72,122,899).  visible(lb2gt_6,899). happensAt(inactive(lb2gt_6),899).
orientation(lb2gt_5,152,900). coords(lb2gt_5,70,113,900).  visible(lb2gt_5,900). happensAt(walking(lb2gt_5),900). orientation(lb2gt_6,151,900). coords(lb2gt_6,72,122,900).  visible(lb2gt_6,900). happensAt(inactive(lb2gt_6),900).
orientation(lb2gt_5,152,901). coords(lb2gt_5,71,113,901).  visible(lb2gt_5,901). happensAt(walking(lb2gt_5),901). orientation(lb2gt_6,151,901). coords(lb2gt_6,71,122,901).  visible(lb2gt_6,901). happensAt(inactive(lb2gt_6),901).
orientation(lb2gt_5,152,902). coords(lb2gt_5,71,112,902).  visible(lb2gt_5,902). happensAt(walking(lb2gt_5),902). orientation(lb2gt_6,151,902). coords(lb2gt_6,71,122,902).  visible(lb2gt_6,902). happensAt(inactive(lb2gt_6),902).
orientation(lb2gt_5,152,903). coords(lb2gt_5,71,113,903).  visible(lb2gt_5,903). happensAt(walking(lb2gt_5),903). orientation(lb2gt_6,151,903). coords(lb2gt_6,71,122,903).  visible(lb2gt_6,903). happensAt(inactive(lb2gt_6),903).
#show holdsAt/2.