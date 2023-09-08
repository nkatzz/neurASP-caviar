
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

	orientation(wk2gt_1,127,848). coords(wk2gt_1,115,53,848).  visible(wk2gt_1,848). happensAt(inactive(wk2gt_1),848). orientation(wk2gt_5,156,848). coords(wk2gt_5,41,106,848).  visible(wk2gt_5,848). happensAt(walking(wk2gt_5),848).
orientation(wk2gt_1,127,849). coords(wk2gt_1,115,53,849).  visible(wk2gt_1,849). happensAt(inactive(wk2gt_1),849). orientation(wk2gt_5,156,849). coords(wk2gt_5,41,106,849).  visible(wk2gt_5,849). happensAt(walking(wk2gt_5),849).
orientation(wk2gt_1,127,850). coords(wk2gt_1,115,53,850).  visible(wk2gt_1,850). happensAt(inactive(wk2gt_1),850). orientation(wk2gt_5,156,850). coords(wk2gt_5,41,106,850).  visible(wk2gt_5,850). happensAt(walking(wk2gt_5),850).
orientation(wk2gt_1,127,851). coords(wk2gt_1,115,53,851).  visible(wk2gt_1,851). happensAt(inactive(wk2gt_1),851). orientation(wk2gt_5,156,851). coords(wk2gt_5,41,106,851).  visible(wk2gt_5,851). happensAt(walking(wk2gt_5),851).
orientation(wk2gt_1,127,852). coords(wk2gt_1,115,53,852).  visible(wk2gt_1,852). happensAt(inactive(wk2gt_1),852). orientation(wk2gt_5,156,852). coords(wk2gt_5,42,105,852).  visible(wk2gt_5,852). happensAt(walking(wk2gt_5),852).
orientation(wk2gt_1,127,853). coords(wk2gt_1,115,53,853).  visible(wk2gt_1,853). happensAt(inactive(wk2gt_1),853). orientation(wk2gt_5,156,853). coords(wk2gt_5,43,105,853).  visible(wk2gt_5,853). happensAt(walking(wk2gt_5),853).
orientation(wk2gt_1,127,854). coords(wk2gt_1,115,53,854).  visible(wk2gt_1,854). happensAt(inactive(wk2gt_1),854). orientation(wk2gt_5,156,854). coords(wk2gt_5,43,105,854).  visible(wk2gt_5,854). happensAt(walking(wk2gt_5),854).
orientation(wk2gt_1,127,855). coords(wk2gt_1,115,53,855).  visible(wk2gt_1,855). happensAt(inactive(wk2gt_1),855). orientation(wk2gt_5,156,855). coords(wk2gt_5,43,105,855).  visible(wk2gt_5,855). happensAt(walking(wk2gt_5),855).
orientation(wk2gt_1,127,856). coords(wk2gt_1,115,53,856).  visible(wk2gt_1,856). happensAt(inactive(wk2gt_1),856). orientation(wk2gt_5,156,856). coords(wk2gt_5,43,105,856).  visible(wk2gt_5,856). happensAt(walking(wk2gt_5),856).
orientation(wk2gt_1,127,857). coords(wk2gt_1,115,53,857).  visible(wk2gt_1,857). happensAt(inactive(wk2gt_1),857). orientation(wk2gt_5,156,857). coords(wk2gt_5,43,105,857).  visible(wk2gt_5,857). happensAt(walking(wk2gt_5),857).
orientation(wk2gt_1,127,858). coords(wk2gt_1,115,53,858).  visible(wk2gt_1,858). happensAt(inactive(wk2gt_1),858). orientation(wk2gt_5,156,858). coords(wk2gt_5,44,105,858).  visible(wk2gt_5,858). happensAt(walking(wk2gt_5),858).
orientation(wk2gt_1,127,859). coords(wk2gt_1,115,53,859).  visible(wk2gt_1,859). happensAt(inactive(wk2gt_1),859). orientation(wk2gt_5,156,859). coords(wk2gt_5,44,105,859).  visible(wk2gt_5,859). happensAt(walking(wk2gt_5),859).
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
#show holdsAt/2.