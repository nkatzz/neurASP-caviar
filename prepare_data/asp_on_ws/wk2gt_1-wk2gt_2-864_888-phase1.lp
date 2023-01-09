
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

	orientation(wk2gt_1,127,864). coords(wk2gt_1,115,53,864).  visible(wk2gt_1,864). happensAt(inactive(wk2gt_1),864). orientation(wk2gt_2,142,864). coords(wk2gt_2,99,63,864).  visible(wk2gt_2,864). happensAt(inactive(wk2gt_2),864).
orientation(wk2gt_1,127,865). coords(wk2gt_1,115,53,865).  visible(wk2gt_1,865). happensAt(inactive(wk2gt_1),865). orientation(wk2gt_2,142,865). coords(wk2gt_2,99,63,865).  visible(wk2gt_2,865). happensAt(inactive(wk2gt_2),865).
orientation(wk2gt_1,127,866). coords(wk2gt_1,115,53,866).  visible(wk2gt_1,866). happensAt(inactive(wk2gt_1),866). orientation(wk2gt_2,142,866). coords(wk2gt_2,99,63,866).  visible(wk2gt_2,866). happensAt(inactive(wk2gt_2),866).
orientation(wk2gt_1,127,867). coords(wk2gt_1,115,53,867).  visible(wk2gt_1,867). happensAt(inactive(wk2gt_1),867). orientation(wk2gt_2,142,867). coords(wk2gt_2,99,63,867).  visible(wk2gt_2,867). happensAt(inactive(wk2gt_2),867).
orientation(wk2gt_1,127,868). coords(wk2gt_1,115,53,868).  visible(wk2gt_1,868). happensAt(inactive(wk2gt_1),868). orientation(wk2gt_2,142,868). coords(wk2gt_2,99,63,868).  visible(wk2gt_2,868). happensAt(inactive(wk2gt_2),868).
orientation(wk2gt_1,127,869). coords(wk2gt_1,115,53,869).  visible(wk2gt_1,869). happensAt(inactive(wk2gt_1),869). orientation(wk2gt_2,142,869). coords(wk2gt_2,99,63,869).  visible(wk2gt_2,869). happensAt(inactive(wk2gt_2),869).
orientation(wk2gt_1,127,870). coords(wk2gt_1,115,53,870).  visible(wk2gt_1,870). happensAt(inactive(wk2gt_1),870). orientation(wk2gt_2,142,870). coords(wk2gt_2,99,63,870).  visible(wk2gt_2,870). happensAt(inactive(wk2gt_2),870).
orientation(wk2gt_1,127,871). coords(wk2gt_1,115,53,871).  visible(wk2gt_1,871). happensAt(inactive(wk2gt_1),871). orientation(wk2gt_2,142,871). coords(wk2gt_2,99,63,871).  visible(wk2gt_2,871). happensAt(inactive(wk2gt_2),871).
orientation(wk2gt_1,127,872). coords(wk2gt_1,115,53,872).  visible(wk2gt_1,872). happensAt(inactive(wk2gt_1),872). orientation(wk2gt_2,142,872). coords(wk2gt_2,99,63,872).  visible(wk2gt_2,872). happensAt(inactive(wk2gt_2),872).
orientation(wk2gt_1,127,873). coords(wk2gt_1,115,53,873).  visible(wk2gt_1,873). happensAt(inactive(wk2gt_1),873). orientation(wk2gt_2,142,873). coords(wk2gt_2,99,63,873).  visible(wk2gt_2,873). happensAt(inactive(wk2gt_2),873).
orientation(wk2gt_1,127,874). coords(wk2gt_1,115,53,874).  visible(wk2gt_1,874). happensAt(inactive(wk2gt_1),874). orientation(wk2gt_2,142,874). coords(wk2gt_2,99,63,874).  visible(wk2gt_2,874). happensAt(inactive(wk2gt_2),874).
orientation(wk2gt_1,127,875). coords(wk2gt_1,115,53,875).  visible(wk2gt_1,875). happensAt(inactive(wk2gt_1),875). orientation(wk2gt_2,142,875). coords(wk2gt_2,99,63,875).  visible(wk2gt_2,875). happensAt(inactive(wk2gt_2),875).
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
#show holdsAt/2.