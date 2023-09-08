
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

	orientation(br1gt_2,12,850). coords(br1gt_2,86,209,850).  visible(br1gt_2,850). happensAt(inactive(br1gt_2),850). orientation(br1gt_3,157,850). coords(br1gt_3,278,196,850).  visible(br1gt_3,850). happensAt(walking(br1gt_3),850).
orientation(br1gt_2,12,851). coords(br1gt_2,86,209,851).  visible(br1gt_2,851). happensAt(inactive(br1gt_2),851). orientation(br1gt_3,150,851). coords(br1gt_3,278,194,851).  visible(br1gt_3,851). happensAt(walking(br1gt_3),851).
orientation(br1gt_2,12,852). coords(br1gt_2,86,209,852).  visible(br1gt_2,852). happensAt(inactive(br1gt_2),852). orientation(br1gt_3,150,852). coords(br1gt_3,281,192,852).  visible(br1gt_3,852). happensAt(walking(br1gt_3),852).
orientation(br1gt_2,12,853). coords(br1gt_2,85,208,853).  visible(br1gt_2,853). happensAt(inactive(br1gt_2),853). orientation(br1gt_3,150,853). coords(br1gt_3,282,190,853).  visible(br1gt_3,853). happensAt(walking(br1gt_3),853).
orientation(br1gt_2,12,854). coords(br1gt_2,85,208,854).  visible(br1gt_2,854). happensAt(inactive(br1gt_2),854). orientation(br1gt_3,145,854). coords(br1gt_3,284,188,854).  visible(br1gt_3,854). happensAt(walking(br1gt_3),854).
orientation(br1gt_2,12,855). coords(br1gt_2,85,208,855).  visible(br1gt_2,855). happensAt(inactive(br1gt_2),855). orientation(br1gt_3,145,855). coords(br1gt_3,285,187,855).  visible(br1gt_3,855). happensAt(walking(br1gt_3),855).
orientation(br1gt_2,12,856). coords(br1gt_2,85,208,856).  visible(br1gt_2,856). happensAt(inactive(br1gt_2),856). orientation(br1gt_3,145,856). coords(br1gt_3,287,187,856).  visible(br1gt_3,856). happensAt(walking(br1gt_3),856).
orientation(br1gt_2,12,857). coords(br1gt_2,85,208,857).  visible(br1gt_2,857). happensAt(inactive(br1gt_2),857). orientation(br1gt_3,145,857). coords(br1gt_3,287,187,857).  visible(br1gt_3,857). happensAt(walking(br1gt_3),857).
orientation(br1gt_2,12,858). coords(br1gt_2,85,208,858).  visible(br1gt_2,858). happensAt(inactive(br1gt_2),858). orientation(br1gt_3,145,858). coords(br1gt_3,287,186,858).  visible(br1gt_3,858). happensAt(walking(br1gt_3),858).
orientation(br1gt_2,12,859). coords(br1gt_2,85,208,859).  visible(br1gt_2,859). happensAt(inactive(br1gt_2),859). orientation(br1gt_3,145,859). coords(br1gt_3,287,185,859).  visible(br1gt_3,859). happensAt(walking(br1gt_3),859).
orientation(br1gt_2,12,860). coords(br1gt_2,85,208,860).  visible(br1gt_2,860). happensAt(inactive(br1gt_2),860). orientation(br1gt_3,145,860). coords(br1gt_3,287,185,860).  visible(br1gt_3,860). happensAt(walking(br1gt_3),860).
orientation(br1gt_2,12,861). coords(br1gt_2,85,208,861).  visible(br1gt_2,861). happensAt(inactive(br1gt_2),861). orientation(br1gt_3,154,861). coords(br1gt_3,288,182,861).  visible(br1gt_3,861). happensAt(walking(br1gt_3),861).
orientation(br1gt_2,12,862). coords(br1gt_2,85,208,862).  visible(br1gt_2,862). happensAt(inactive(br1gt_2),862). orientation(br1gt_3,152,862). coords(br1gt_3,289,182,862).  visible(br1gt_3,862). happensAt(walking(br1gt_3),862).
orientation(br1gt_2,12,863). coords(br1gt_2,85,208,863).  visible(br1gt_2,863). happensAt(inactive(br1gt_2),863). orientation(br1gt_3,152,863). coords(br1gt_3,289,182,863).  visible(br1gt_3,863). happensAt(walking(br1gt_3),863).
orientation(br1gt_2,12,864). coords(br1gt_2,85,208,864).  visible(br1gt_2,864). happensAt(inactive(br1gt_2),864). orientation(br1gt_3,152,864). coords(br1gt_3,289,182,864).  visible(br1gt_3,864). happensAt(walking(br1gt_3),864).
orientation(br1gt_2,12,865). coords(br1gt_2,85,205,865).  visible(br1gt_2,865). happensAt(inactive(br1gt_2),865). orientation(br1gt_3,152,865). coords(br1gt_3,289,181,865).  visible(br1gt_3,865). happensAt(walking(br1gt_3),865).
orientation(br1gt_2,12,866). coords(br1gt_2,85,208,866).  visible(br1gt_2,866). happensAt(inactive(br1gt_2),866). orientation(br1gt_3,145,866). coords(br1gt_3,289,180,866).  visible(br1gt_3,866). happensAt(walking(br1gt_3),866).
orientation(br1gt_2,12,867). coords(br1gt_2,85,208,867).  visible(br1gt_2,867). happensAt(inactive(br1gt_2),867). orientation(br1gt_3,145,867). coords(br1gt_3,289,178,867).  visible(br1gt_3,867). happensAt(walking(br1gt_3),867).
orientation(br1gt_2,12,868). coords(br1gt_2,85,207,868).  visible(br1gt_2,868). happensAt(inactive(br1gt_2),868). orientation(br1gt_3,104,868). coords(br1gt_3,289,176,868).  visible(br1gt_3,868). happensAt(walking(br1gt_3),868).
orientation(br1gt_2,12,869). coords(br1gt_2,85,207,869).  visible(br1gt_2,869). happensAt(inactive(br1gt_2),869). orientation(br1gt_3,104,869). coords(br1gt_3,289,174,869).  visible(br1gt_3,869). happensAt(walking(br1gt_3),869).
orientation(br1gt_2,5,870). coords(br1gt_2,85,207,870).  visible(br1gt_2,870). happensAt(inactive(br1gt_2),870). orientation(br1gt_3,104,870). coords(br1gt_3,290,173,870).  visible(br1gt_3,870). happensAt(walking(br1gt_3),870).
orientation(br1gt_2,5,871). coords(br1gt_2,85,207,871).  visible(br1gt_2,871). happensAt(inactive(br1gt_2),871). orientation(br1gt_3,104,871). coords(br1gt_3,291,173,871).  visible(br1gt_3,871). happensAt(walking(br1gt_3),871).
orientation(br1gt_2,5,872). coords(br1gt_2,85,206,872).  visible(br1gt_2,872). happensAt(inactive(br1gt_2),872). orientation(br1gt_3,104,872). coords(br1gt_3,292,172,872).  visible(br1gt_3,872). happensAt(walking(br1gt_3),872).
orientation(br1gt_2,5,873). coords(br1gt_2,85,206,873).  visible(br1gt_2,873). happensAt(inactive(br1gt_2),873). orientation(br1gt_3,104,873). coords(br1gt_3,293,171,873).  visible(br1gt_3,873). happensAt(walking(br1gt_3),873).
orientation(br1gt_2,5,874). coords(br1gt_2,85,206,874).  visible(br1gt_2,874). happensAt(inactive(br1gt_2),874). orientation(br1gt_3,104,874). coords(br1gt_3,293,170,874).  visible(br1gt_3,874). happensAt(walking(br1gt_3),874).
#show holdsAt/2.