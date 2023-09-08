
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

	orientation(lbbcgt_3,163,877). coords(lbbcgt_3,41,136,877).  visible(lbbcgt_3,877). happensAt(walking(lbbcgt_3),877). orientation(lbbcgt_4,137,877). coords(lbbcgt_4,8,63,877).  visible(lbbcgt_4,877). happensAt(walking(lbbcgt_4),877).
orientation(lbbcgt_3,163,878). coords(lbbcgt_3,41,137,878).  visible(lbbcgt_3,878). happensAt(walking(lbbcgt_3),878). orientation(lbbcgt_4,137,878). coords(lbbcgt_4,9,63,878).  visible(lbbcgt_4,878). happensAt(walking(lbbcgt_4),878).
orientation(lbbcgt_3,163,879). coords(lbbcgt_3,42,137,879).  visible(lbbcgt_3,879). happensAt(walking(lbbcgt_3),879). orientation(lbbcgt_4,137,879). coords(lbbcgt_4,9,63,879).  visible(lbbcgt_4,879). happensAt(walking(lbbcgt_4),879).
orientation(lbbcgt_3,163,880). coords(lbbcgt_3,42,138,880).  visible(lbbcgt_3,880). happensAt(walking(lbbcgt_3),880). orientation(lbbcgt_4,137,880). coords(lbbcgt_4,9,63,880).  visible(lbbcgt_4,880). happensAt(walking(lbbcgt_4),880).
orientation(lbbcgt_3,163,881). coords(lbbcgt_3,42,139,881).  visible(lbbcgt_3,881). happensAt(walking(lbbcgt_3),881). orientation(lbbcgt_4,137,881). coords(lbbcgt_4,9,62,881).  visible(lbbcgt_4,881). happensAt(walking(lbbcgt_4),881).
orientation(lbbcgt_3,163,882). coords(lbbcgt_3,42,139,882).  visible(lbbcgt_3,882). happensAt(walking(lbbcgt_3),882). orientation(lbbcgt_4,137,882). coords(lbbcgt_4,9,61,882).  visible(lbbcgt_4,882). happensAt(walking(lbbcgt_4),882).
orientation(lbbcgt_3,163,883). coords(lbbcgt_3,41,139,883).  visible(lbbcgt_3,883). happensAt(walking(lbbcgt_3),883). orientation(lbbcgt_4,132,883). coords(lbbcgt_4,9,61,883).  visible(lbbcgt_4,883). happensAt(walking(lbbcgt_4),883).
orientation(lbbcgt_3,163,884). coords(lbbcgt_3,41,140,884).  visible(lbbcgt_3,884). happensAt(walking(lbbcgt_3),884). orientation(lbbcgt_4,132,884). coords(lbbcgt_4,10,60,884).  visible(lbbcgt_4,884). happensAt(walking(lbbcgt_4),884).
orientation(lbbcgt_3,163,885). coords(lbbcgt_3,41,140,885).  visible(lbbcgt_3,885). happensAt(walking(lbbcgt_3),885). orientation(lbbcgt_4,132,885). coords(lbbcgt_4,13,59,885).  visible(lbbcgt_4,885). happensAt(walking(lbbcgt_4),885).
orientation(lbbcgt_3,163,886). coords(lbbcgt_3,39,142,886).  visible(lbbcgt_3,886). happensAt(walking(lbbcgt_3),886). orientation(lbbcgt_4,132,886). coords(lbbcgt_4,13,57,886).  visible(lbbcgt_4,886). happensAt(walking(lbbcgt_4),886).
orientation(lbbcgt_3,163,887). coords(lbbcgt_3,39,142,887).  visible(lbbcgt_3,887). happensAt(walking(lbbcgt_3),887). orientation(lbbcgt_4,137,887). coords(lbbcgt_4,13,56,887).  visible(lbbcgt_4,887). happensAt(walking(lbbcgt_4),887).
orientation(lbbcgt_3,163,888). coords(lbbcgt_3,38,144,888).  visible(lbbcgt_3,888). happensAt(walking(lbbcgt_3),888). orientation(lbbcgt_4,137,888). coords(lbbcgt_4,13,55,888).  visible(lbbcgt_4,888). happensAt(walking(lbbcgt_4),888).
orientation(lbbcgt_3,163,889). coords(lbbcgt_3,39,144,889).  visible(lbbcgt_3,889). happensAt(walking(lbbcgt_3),889). orientation(lbbcgt_4,137,889). coords(lbbcgt_4,13,54,889).  visible(lbbcgt_4,889). happensAt(walking(lbbcgt_4),889).
orientation(lbbcgt_3,163,890). coords(lbbcgt_3,38,144,890).  visible(lbbcgt_3,890). happensAt(walking(lbbcgt_3),890). orientation(lbbcgt_4,137,890). coords(lbbcgt_4,13,54,890).  visible(lbbcgt_4,890). happensAt(walking(lbbcgt_4),890).
orientation(lbbcgt_3,163,891). coords(lbbcgt_3,39,146,891).  visible(lbbcgt_3,891). happensAt(walking(lbbcgt_3),891). orientation(lbbcgt_4,137,891). coords(lbbcgt_4,13,53,891).  happensAt(occluded(lbbcgt_4),891). happensAt(walking(lbbcgt_4),891).
orientation(lbbcgt_3,163,892). coords(lbbcgt_3,39,148,892).  visible(lbbcgt_3,892). happensAt(walking(lbbcgt_3),892). orientation(lbbcgt_4,137,892). coords(lbbcgt_4,13,53,892).  happensAt(occluded(lbbcgt_4),892). happensAt(walking(lbbcgt_4),892).
orientation(lbbcgt_3,163,893). coords(lbbcgt_3,39,148,893).  visible(lbbcgt_3,893). happensAt(walking(lbbcgt_3),893). orientation(lbbcgt_4,137,893). coords(lbbcgt_4,13,53,893).  happensAt(occluded(lbbcgt_4),893). happensAt(walking(lbbcgt_4),893).
orientation(lbbcgt_3,163,894). coords(lbbcgt_3,39,149,894).  visible(lbbcgt_3,894). happensAt(walking(lbbcgt_3),894). orientation(lbbcgt_4,137,894). coords(lbbcgt_4,13,53,894).  happensAt(occluded(lbbcgt_4),894). happensAt(walking(lbbcgt_4),894).
orientation(lbbcgt_3,163,895). coords(lbbcgt_3,39,149,895).  visible(lbbcgt_3,895). happensAt(walking(lbbcgt_3),895). orientation(lbbcgt_4,137,895). coords(lbbcgt_4,17,46,895).  visible(lbbcgt_4,895). happensAt(walking(lbbcgt_4),895).
orientation(lbbcgt_3,163,896). coords(lbbcgt_3,39,149,896).  visible(lbbcgt_3,896). happensAt(walking(lbbcgt_3),896). orientation(lbbcgt_4,140,896). coords(lbbcgt_4,17,46,896).  visible(lbbcgt_4,896). happensAt(walking(lbbcgt_4),896).
orientation(lbbcgt_3,163,897). coords(lbbcgt_3,39,150,897).  visible(lbbcgt_3,897). happensAt(walking(lbbcgt_3),897). orientation(lbbcgt_4,144,897). coords(lbbcgt_4,17,45,897).  visible(lbbcgt_4,897). happensAt(walking(lbbcgt_4),897).
orientation(lbbcgt_3,163,898). coords(lbbcgt_3,39,151,898).  visible(lbbcgt_3,898). happensAt(walking(lbbcgt_3),898). orientation(lbbcgt_4,146,898). coords(lbbcgt_4,17,45,898).  visible(lbbcgt_4,898). happensAt(walking(lbbcgt_4),898).
orientation(lbbcgt_3,163,899). coords(lbbcgt_3,39,151,899).  visible(lbbcgt_3,899). happensAt(walking(lbbcgt_3),899). orientation(lbbcgt_4,146,899). coords(lbbcgt_4,16,45,899).  visible(lbbcgt_4,899). happensAt(walking(lbbcgt_4),899).
orientation(lbbcgt_3,163,900). coords(lbbcgt_3,39,151,900).  visible(lbbcgt_3,900). happensAt(walking(lbbcgt_3),900). orientation(lbbcgt_4,146,900). coords(lbbcgt_4,16,45,900).  visible(lbbcgt_4,900). happensAt(walking(lbbcgt_4),900).
orientation(lbbcgt_3,163,901). coords(lbbcgt_3,39,152,901).  visible(lbbcgt_3,901). happensAt(walking(lbbcgt_3),901). orientation(lbbcgt_4,146,901). coords(lbbcgt_4,18,45,901).  visible(lbbcgt_4,901). happensAt(walking(lbbcgt_4),901).
#show holdsAt/2.