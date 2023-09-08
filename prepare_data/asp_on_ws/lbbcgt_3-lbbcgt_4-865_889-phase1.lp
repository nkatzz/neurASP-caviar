
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

	orientation(lbbcgt_3,158,865). coords(lbbcgt_3,44,129,865).  visible(lbbcgt_3,865). happensAt(walking(lbbcgt_3),865). orientation(lbbcgt_4,137,865). coords(lbbcgt_4,8,76,865).  visible(lbbcgt_4,865). happensAt(walking(lbbcgt_4),865).
orientation(lbbcgt_3,163,866). coords(lbbcgt_3,43,129,866).  visible(lbbcgt_3,866). happensAt(walking(lbbcgt_3),866). orientation(lbbcgt_4,137,866). coords(lbbcgt_4,7,75,866).  visible(lbbcgt_4,866). happensAt(walking(lbbcgt_4),866).
orientation(lbbcgt_3,163,867). coords(lbbcgt_3,43,129,867).  visible(lbbcgt_3,867). happensAt(walking(lbbcgt_3),867). orientation(lbbcgt_4,142,867). coords(lbbcgt_4,8,75,867).  visible(lbbcgt_4,867). happensAt(walking(lbbcgt_4),867).
orientation(lbbcgt_3,163,868). coords(lbbcgt_3,43,129,868).  visible(lbbcgt_3,868). happensAt(walking(lbbcgt_3),868). orientation(lbbcgt_4,142,868). coords(lbbcgt_4,9,74,868).  visible(lbbcgt_4,868). happensAt(walking(lbbcgt_4),868).
orientation(lbbcgt_3,163,869). coords(lbbcgt_3,43,130,869).  visible(lbbcgt_3,869). happensAt(walking(lbbcgt_3),869). orientation(lbbcgt_4,142,869). coords(lbbcgt_4,9,74,869).  visible(lbbcgt_4,869). happensAt(walking(lbbcgt_4),869).
orientation(lbbcgt_3,163,870). coords(lbbcgt_3,43,130,870).  visible(lbbcgt_3,870). happensAt(walking(lbbcgt_3),870). orientation(lbbcgt_4,142,870). coords(lbbcgt_4,9,72,870).  visible(lbbcgt_4,870). happensAt(walking(lbbcgt_4),870).
orientation(lbbcgt_3,163,871). coords(lbbcgt_3,43,131,871).  visible(lbbcgt_3,871). happensAt(walking(lbbcgt_3),871). orientation(lbbcgt_4,142,871). coords(lbbcgt_4,9,71,871).  visible(lbbcgt_4,871). happensAt(walking(lbbcgt_4),871).
orientation(lbbcgt_3,163,872). coords(lbbcgt_3,42,131,872).  visible(lbbcgt_3,872). happensAt(walking(lbbcgt_3),872). orientation(lbbcgt_4,142,872). coords(lbbcgt_4,9,70,872).  visible(lbbcgt_4,872). happensAt(walking(lbbcgt_4),872).
orientation(lbbcgt_3,163,873). coords(lbbcgt_3,42,131,873).  visible(lbbcgt_3,873). happensAt(walking(lbbcgt_3),873). orientation(lbbcgt_4,142,873). coords(lbbcgt_4,8,70,873).  visible(lbbcgt_4,873). happensAt(walking(lbbcgt_4),873).
orientation(lbbcgt_3,163,874). coords(lbbcgt_3,41,132,874).  visible(lbbcgt_3,874). happensAt(walking(lbbcgt_3),874). orientation(lbbcgt_4,142,874). coords(lbbcgt_4,8,69,874).  visible(lbbcgt_4,874). happensAt(walking(lbbcgt_4),874).
orientation(lbbcgt_3,163,875). coords(lbbcgt_3,42,135,875).  visible(lbbcgt_3,875). happensAt(walking(lbbcgt_3),875). orientation(lbbcgt_4,142,875). coords(lbbcgt_4,8,67,875).  visible(lbbcgt_4,875). happensAt(walking(lbbcgt_4),875).
orientation(lbbcgt_3,163,876). coords(lbbcgt_3,42,135,876).  visible(lbbcgt_3,876). happensAt(walking(lbbcgt_3),876). orientation(lbbcgt_4,142,876). coords(lbbcgt_4,8,66,876).  visible(lbbcgt_4,876). happensAt(walking(lbbcgt_4),876).
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
#show holdsAt/2.