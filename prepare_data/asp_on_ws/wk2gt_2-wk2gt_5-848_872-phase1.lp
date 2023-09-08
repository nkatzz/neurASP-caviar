
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

	orientation(wk2gt_2,142,848). coords(wk2gt_2,99,63,848).  visible(wk2gt_2,848). happensAt(inactive(wk2gt_2),848). orientation(wk2gt_5,156,848). coords(wk2gt_5,41,106,848).  visible(wk2gt_5,848). happensAt(walking(wk2gt_5),848).
orientation(wk2gt_2,142,849). coords(wk2gt_2,99,63,849).  visible(wk2gt_2,849). happensAt(inactive(wk2gt_2),849). orientation(wk2gt_5,156,849). coords(wk2gt_5,41,106,849).  visible(wk2gt_5,849). happensAt(walking(wk2gt_5),849).
orientation(wk2gt_2,142,850). coords(wk2gt_2,99,63,850).  visible(wk2gt_2,850). happensAt(inactive(wk2gt_2),850). orientation(wk2gt_5,156,850). coords(wk2gt_5,41,106,850).  visible(wk2gt_5,850). happensAt(walking(wk2gt_5),850).
orientation(wk2gt_2,142,851). coords(wk2gt_2,99,63,851).  visible(wk2gt_2,851). happensAt(inactive(wk2gt_2),851). orientation(wk2gt_5,156,851). coords(wk2gt_5,41,106,851).  visible(wk2gt_5,851). happensAt(walking(wk2gt_5),851).
orientation(wk2gt_2,142,852). coords(wk2gt_2,99,63,852).  visible(wk2gt_2,852). happensAt(inactive(wk2gt_2),852). orientation(wk2gt_5,156,852). coords(wk2gt_5,42,105,852).  visible(wk2gt_5,852). happensAt(walking(wk2gt_5),852).
orientation(wk2gt_2,142,853). coords(wk2gt_2,99,63,853).  visible(wk2gt_2,853). happensAt(inactive(wk2gt_2),853). orientation(wk2gt_5,156,853). coords(wk2gt_5,43,105,853).  visible(wk2gt_5,853). happensAt(walking(wk2gt_5),853).
orientation(wk2gt_2,142,854). coords(wk2gt_2,99,63,854).  visible(wk2gt_2,854). happensAt(inactive(wk2gt_2),854). orientation(wk2gt_5,156,854). coords(wk2gt_5,43,105,854).  visible(wk2gt_5,854). happensAt(walking(wk2gt_5),854).
orientation(wk2gt_2,142,855). coords(wk2gt_2,99,63,855).  visible(wk2gt_2,855). happensAt(inactive(wk2gt_2),855). orientation(wk2gt_5,156,855). coords(wk2gt_5,43,105,855).  visible(wk2gt_5,855). happensAt(walking(wk2gt_5),855).
orientation(wk2gt_2,142,856). coords(wk2gt_2,99,63,856).  visible(wk2gt_2,856). happensAt(inactive(wk2gt_2),856). orientation(wk2gt_5,156,856). coords(wk2gt_5,43,105,856).  visible(wk2gt_5,856). happensAt(walking(wk2gt_5),856).
orientation(wk2gt_2,142,857). coords(wk2gt_2,99,63,857).  visible(wk2gt_2,857). happensAt(inactive(wk2gt_2),857). orientation(wk2gt_5,156,857). coords(wk2gt_5,43,105,857).  visible(wk2gt_5,857). happensAt(walking(wk2gt_5),857).
orientation(wk2gt_2,142,858). coords(wk2gt_2,99,63,858).  visible(wk2gt_2,858). happensAt(inactive(wk2gt_2),858). orientation(wk2gt_5,156,858). coords(wk2gt_5,44,105,858).  visible(wk2gt_5,858). happensAt(walking(wk2gt_5),858).
orientation(wk2gt_2,142,859). coords(wk2gt_2,99,63,859).  visible(wk2gt_2,859). happensAt(inactive(wk2gt_2),859). orientation(wk2gt_5,156,859). coords(wk2gt_5,44,105,859).  visible(wk2gt_5,859). happensAt(walking(wk2gt_5),859).
orientation(wk2gt_2,142,860). coords(wk2gt_2,99,63,860).  visible(wk2gt_2,860). happensAt(inactive(wk2gt_2),860). orientation(wk2gt_5,156,860). coords(wk2gt_5,44,104,860).  visible(wk2gt_5,860). happensAt(walking(wk2gt_5),860).
orientation(wk2gt_2,142,861). coords(wk2gt_2,99,63,861).  visible(wk2gt_2,861). happensAt(inactive(wk2gt_2),861). orientation(wk2gt_5,156,861). coords(wk2gt_5,44,103,861).  visible(wk2gt_5,861). happensAt(walking(wk2gt_5),861).
orientation(wk2gt_2,142,862). coords(wk2gt_2,99,63,862).  visible(wk2gt_2,862). happensAt(inactive(wk2gt_2),862). orientation(wk2gt_5,156,862). coords(wk2gt_5,44,103,862).  visible(wk2gt_5,862). happensAt(walking(wk2gt_5),862).
orientation(wk2gt_2,142,863). coords(wk2gt_2,99,63,863).  visible(wk2gt_2,863). happensAt(inactive(wk2gt_2),863). orientation(wk2gt_5,156,863). coords(wk2gt_5,45,102,863).  visible(wk2gt_5,863). happensAt(walking(wk2gt_5),863).
orientation(wk2gt_2,142,864). coords(wk2gt_2,99,63,864).  visible(wk2gt_2,864). happensAt(inactive(wk2gt_2),864). orientation(wk2gt_5,156,864). coords(wk2gt_5,45,101,864).  visible(wk2gt_5,864). happensAt(walking(wk2gt_5),864).
orientation(wk2gt_2,142,865). coords(wk2gt_2,99,63,865).  visible(wk2gt_2,865). happensAt(inactive(wk2gt_2),865). orientation(wk2gt_5,156,865). coords(wk2gt_5,46,101,865).  visible(wk2gt_5,865). happensAt(walking(wk2gt_5),865).
orientation(wk2gt_2,142,866). coords(wk2gt_2,99,63,866).  visible(wk2gt_2,866). happensAt(inactive(wk2gt_2),866). orientation(wk2gt_5,156,866). coords(wk2gt_5,45,101,866).  visible(wk2gt_5,866). happensAt(walking(wk2gt_5),866).
orientation(wk2gt_2,142,867). coords(wk2gt_2,99,63,867).  visible(wk2gt_2,867). happensAt(inactive(wk2gt_2),867). orientation(wk2gt_5,156,867). coords(wk2gt_5,47,101,867).  visible(wk2gt_5,867). happensAt(walking(wk2gt_5),867).
orientation(wk2gt_2,142,868). coords(wk2gt_2,99,63,868).  visible(wk2gt_2,868). happensAt(inactive(wk2gt_2),868). orientation(wk2gt_5,156,868). coords(wk2gt_5,48,101,868).  visible(wk2gt_5,868). happensAt(walking(wk2gt_5),868).
orientation(wk2gt_2,142,869). coords(wk2gt_2,99,63,869).  visible(wk2gt_2,869). happensAt(inactive(wk2gt_2),869). orientation(wk2gt_5,156,869). coords(wk2gt_5,48,101,869).  visible(wk2gt_5,869). happensAt(walking(wk2gt_5),869).
orientation(wk2gt_2,142,870). coords(wk2gt_2,99,63,870).  visible(wk2gt_2,870). happensAt(inactive(wk2gt_2),870). orientation(wk2gt_5,156,870). coords(wk2gt_5,48,101,870).  visible(wk2gt_5,870). happensAt(walking(wk2gt_5),870).
orientation(wk2gt_2,142,871). coords(wk2gt_2,99,63,871).  visible(wk2gt_2,871). happensAt(inactive(wk2gt_2),871). orientation(wk2gt_5,156,871). coords(wk2gt_5,48,101,871).  visible(wk2gt_5,871). happensAt(walking(wk2gt_5),871).
orientation(wk2gt_2,142,872). coords(wk2gt_2,99,63,872).  visible(wk2gt_2,872). happensAt(inactive(wk2gt_2),872). orientation(wk2gt_5,156,872). coords(wk2gt_5,48,100,872).  visible(wk2gt_5,872). happensAt(walking(wk2gt_5),872).
#show holdsAt/2.