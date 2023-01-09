
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

	orientation(wk2gt_1,127,860). coords(wk2gt_1,115,53,860).  visible(wk2gt_1,860). happensAt(inactive(wk2gt_1),860). orientation(wk2gt_5,156,860). coords(wk2gt_5,44,104,860).  visible(wk2gt_5,860). happensAt(walking(wk2gt_5),860).
orientation(wk2gt_1,127,861). coords(wk2gt_1,115,53,861).  visible(wk2gt_1,861). happensAt(inactive(wk2gt_1),861). orientation(wk2gt_5,156,861). coords(wk2gt_5,44,103,861).  visible(wk2gt_5,861). happensAt(walking(wk2gt_5),861).
orientation(wk2gt_1,127,862). coords(wk2gt_1,115,53,862).  visible(wk2gt_1,862). happensAt(inactive(wk2gt_1),862). orientation(wk2gt_5,156,862). coords(wk2gt_5,44,103,862).  visible(wk2gt_5,862). happensAt(walking(wk2gt_5),862).
orientation(wk2gt_1,127,863). coords(wk2gt_1,115,53,863).  visible(wk2gt_1,863). happensAt(inactive(wk2gt_1),863). orientation(wk2gt_5,156,863). coords(wk2gt_5,45,102,863).  visible(wk2gt_5,863). happensAt(walking(wk2gt_5),863).
orientation(wk2gt_1,127,864). coords(wk2gt_1,115,53,864).  visible(wk2gt_1,864). happensAt(inactive(wk2gt_1),864). orientation(wk2gt_5,156,864). coords(wk2gt_5,45,101,864).  visible(wk2gt_5,864). happensAt(walking(wk2gt_5),864).
orientation(wk2gt_1,127,865). coords(wk2gt_1,115,53,865).  visible(wk2gt_1,865). happensAt(inactive(wk2gt_1),865). orientation(wk2gt_5,156,865). coords(wk2gt_5,46,101,865).  visible(wk2gt_5,865). happensAt(walking(wk2gt_5),865).
orientation(wk2gt_1,127,866). coords(wk2gt_1,115,53,866).  visible(wk2gt_1,866). happensAt(inactive(wk2gt_1),866). orientation(wk2gt_5,156,866). coords(wk2gt_5,45,101,866).  visible(wk2gt_5,866). happensAt(walking(wk2gt_5),866).
orientation(wk2gt_1,127,867). coords(wk2gt_1,115,53,867).  visible(wk2gt_1,867). happensAt(inactive(wk2gt_1),867). orientation(wk2gt_5,156,867). coords(wk2gt_5,47,101,867).  visible(wk2gt_5,867). happensAt(walking(wk2gt_5),867).
orientation(wk2gt_1,127,868). coords(wk2gt_1,115,53,868).  visible(wk2gt_1,868). happensAt(inactive(wk2gt_1),868). orientation(wk2gt_5,156,868). coords(wk2gt_5,48,101,868).  visible(wk2gt_5,868). happensAt(walking(wk2gt_5),868).
orientation(wk2gt_1,127,869). coords(wk2gt_1,115,53,869).  visible(wk2gt_1,869). happensAt(inactive(wk2gt_1),869). orientation(wk2gt_5,156,869). coords(wk2gt_5,48,101,869).  visible(wk2gt_5,869). happensAt(walking(wk2gt_5),869).
orientation(wk2gt_1,127,870). coords(wk2gt_1,115,53,870).  visible(wk2gt_1,870). happensAt(inactive(wk2gt_1),870). orientation(wk2gt_5,156,870). coords(wk2gt_5,48,101,870).  visible(wk2gt_5,870). happensAt(walking(wk2gt_5),870).
orientation(wk2gt_1,127,871). coords(wk2gt_1,115,53,871).  visible(wk2gt_1,871). happensAt(inactive(wk2gt_1),871). orientation(wk2gt_5,156,871). coords(wk2gt_5,48,101,871).  visible(wk2gt_5,871). happensAt(walking(wk2gt_5),871).
orientation(wk2gt_1,127,872). coords(wk2gt_1,115,53,872).  visible(wk2gt_1,872). happensAt(inactive(wk2gt_1),872). orientation(wk2gt_5,156,872). coords(wk2gt_5,48,100,872).  visible(wk2gt_5,872). happensAt(walking(wk2gt_5),872).
orientation(wk2gt_1,127,873). coords(wk2gt_1,115,53,873).  visible(wk2gt_1,873). happensAt(inactive(wk2gt_1),873). orientation(wk2gt_5,156,873). coords(wk2gt_5,48,100,873).  visible(wk2gt_5,873). happensAt(walking(wk2gt_5),873).
orientation(wk2gt_1,127,874). coords(wk2gt_1,115,53,874).  visible(wk2gt_1,874). happensAt(inactive(wk2gt_1),874). orientation(wk2gt_5,156,874). coords(wk2gt_5,48,100,874).  visible(wk2gt_5,874). happensAt(walking(wk2gt_5),874).
orientation(wk2gt_1,127,875). coords(wk2gt_1,115,53,875).  visible(wk2gt_1,875). happensAt(inactive(wk2gt_1),875). orientation(wk2gt_5,156,875). coords(wk2gt_5,49,100,875).  visible(wk2gt_5,875). happensAt(walking(wk2gt_5),875).
orientation(wk2gt_1,127,876). coords(wk2gt_1,115,53,876).  visible(wk2gt_1,876). happensAt(inactive(wk2gt_1),876). orientation(wk2gt_5,156,876). coords(wk2gt_5,49,100,876).  visible(wk2gt_5,876). happensAt(walking(wk2gt_5),876).
orientation(wk2gt_1,127,877). coords(wk2gt_1,115,53,877).  visible(wk2gt_1,877). happensAt(inactive(wk2gt_1),877). orientation(wk2gt_5,156,877). coords(wk2gt_5,49,100,877).  visible(wk2gt_5,877). happensAt(walking(wk2gt_5),877).
orientation(wk2gt_1,127,878). coords(wk2gt_1,115,53,878).  visible(wk2gt_1,878). happensAt(inactive(wk2gt_1),878). orientation(wk2gt_5,156,878). coords(wk2gt_5,50,100,878).  visible(wk2gt_5,878). happensAt(walking(wk2gt_5),878).
orientation(wk2gt_1,127,879). coords(wk2gt_1,115,53,879).  visible(wk2gt_1,879). happensAt(inactive(wk2gt_1),879). orientation(wk2gt_5,156,879). coords(wk2gt_5,52,100,879).  visible(wk2gt_5,879). happensAt(walking(wk2gt_5),879).
orientation(wk2gt_1,127,880). coords(wk2gt_1,115,53,880).  visible(wk2gt_1,880). happensAt(inactive(wk2gt_1),880). orientation(wk2gt_5,156,880). coords(wk2gt_5,52,100,880).  visible(wk2gt_5,880). happensAt(walking(wk2gt_5),880).
orientation(wk2gt_1,127,881). coords(wk2gt_1,115,53,881).  visible(wk2gt_1,881). happensAt(inactive(wk2gt_1),881). orientation(wk2gt_5,156,881). coords(wk2gt_5,52,100,881).  visible(wk2gt_5,881). happensAt(walking(wk2gt_5),881).
orientation(wk2gt_1,127,882). coords(wk2gt_1,115,53,882).  visible(wk2gt_1,882). happensAt(inactive(wk2gt_1),882). orientation(wk2gt_5,156,882). coords(wk2gt_5,52,99,882).  visible(wk2gt_5,882). happensAt(walking(wk2gt_5),882).
orientation(wk2gt_1,127,883). coords(wk2gt_1,115,53,883).  visible(wk2gt_1,883). happensAt(inactive(wk2gt_1),883). orientation(wk2gt_5,156,883). coords(wk2gt_5,52,99,883).  visible(wk2gt_5,883). happensAt(walking(wk2gt_5),883).
orientation(wk2gt_1,127,884). coords(wk2gt_1,115,53,884).  visible(wk2gt_1,884). happensAt(inactive(wk2gt_1),884). orientation(wk2gt_5,156,884). coords(wk2gt_5,53,99,884).  visible(wk2gt_5,884). happensAt(walking(wk2gt_5),884).
#show holdsAt/2.