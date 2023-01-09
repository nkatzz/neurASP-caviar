
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

	orientation(lb2gt_5,152,843). coords(lb2gt_5,97,82,843).  visible(lb2gt_5,843). happensAt(walking(lb2gt_5),843). orientation(lb2gt_6,151,843). coords(lb2gt_6,72,122,843).  visible(lb2gt_6,843). happensAt(inactive(lb2gt_6),843).
orientation(lb2gt_5,152,844). coords(lb2gt_5,96,83,844).  visible(lb2gt_5,844). happensAt(walking(lb2gt_5),844). orientation(lb2gt_6,151,844). coords(lb2gt_6,72,122,844).  visible(lb2gt_6,844). happensAt(inactive(lb2gt_6),844).
orientation(lb2gt_5,152,845). coords(lb2gt_5,96,83,845).  visible(lb2gt_5,845). happensAt(walking(lb2gt_5),845). orientation(lb2gt_6,151,845). coords(lb2gt_6,72,122,845).  visible(lb2gt_6,845). happensAt(inactive(lb2gt_6),845).
orientation(lb2gt_5,152,846). coords(lb2gt_5,96,85,846).  visible(lb2gt_5,846). happensAt(walking(lb2gt_5),846). orientation(lb2gt_6,151,846). coords(lb2gt_6,72,122,846).  visible(lb2gt_6,846). happensAt(inactive(lb2gt_6),846).
orientation(lb2gt_5,152,847). coords(lb2gt_5,95,86,847).  visible(lb2gt_5,847). happensAt(walking(lb2gt_5),847). orientation(lb2gt_6,151,847). coords(lb2gt_6,72,122,847).  visible(lb2gt_6,847). happensAt(inactive(lb2gt_6),847).
orientation(lb2gt_5,152,848). coords(lb2gt_5,95,87,848).  visible(lb2gt_5,848). happensAt(walking(lb2gt_5),848). orientation(lb2gt_6,151,848). coords(lb2gt_6,72,122,848).  visible(lb2gt_6,848). happensAt(inactive(lb2gt_6),848).
orientation(lb2gt_5,152,849). coords(lb2gt_5,94,89,849).  visible(lb2gt_5,849). happensAt(walking(lb2gt_5),849). orientation(lb2gt_6,151,849). coords(lb2gt_6,72,122,849).  visible(lb2gt_6,849). happensAt(inactive(lb2gt_6),849).
orientation(lb2gt_5,152,850). coords(lb2gt_5,93,89,850).  visible(lb2gt_5,850). happensAt(walking(lb2gt_5),850). orientation(lb2gt_6,151,850). coords(lb2gt_6,72,122,850).  visible(lb2gt_6,850). happensAt(inactive(lb2gt_6),850).
orientation(lb2gt_5,152,851). coords(lb2gt_5,92,90,851).  visible(lb2gt_5,851). happensAt(walking(lb2gt_5),851). orientation(lb2gt_6,151,851). coords(lb2gt_6,72,122,851).  visible(lb2gt_6,851). happensAt(inactive(lb2gt_6),851).
orientation(lb2gt_5,152,852). coords(lb2gt_5,91,90,852).  visible(lb2gt_5,852). happensAt(walking(lb2gt_5),852). orientation(lb2gt_6,151,852). coords(lb2gt_6,72,122,852).  visible(lb2gt_6,852). happensAt(inactive(lb2gt_6),852).
orientation(lb2gt_5,152,853). coords(lb2gt_5,90,90,853).  visible(lb2gt_5,853). happensAt(walking(lb2gt_5),853). orientation(lb2gt_6,151,853). coords(lb2gt_6,72,122,853).  visible(lb2gt_6,853). happensAt(inactive(lb2gt_6),853).
orientation(lb2gt_5,152,854). coords(lb2gt_5,88,91,854).  visible(lb2gt_5,854). happensAt(walking(lb2gt_5),854). orientation(lb2gt_6,151,854). coords(lb2gt_6,72,122,854).  visible(lb2gt_6,854). happensAt(inactive(lb2gt_6),854).
orientation(lb2gt_5,152,855). coords(lb2gt_5,87,91,855).  visible(lb2gt_5,855). happensAt(walking(lb2gt_5),855). orientation(lb2gt_6,151,855). coords(lb2gt_6,72,122,855).  visible(lb2gt_6,855). happensAt(inactive(lb2gt_6),855).
orientation(lb2gt_5,152,856). coords(lb2gt_5,86,92,856).  visible(lb2gt_5,856). happensAt(walking(lb2gt_5),856). orientation(lb2gt_6,151,856). coords(lb2gt_6,72,122,856).  visible(lb2gt_6,856). happensAt(inactive(lb2gt_6),856).
orientation(lb2gt_5,152,857). coords(lb2gt_5,85,93,857).  visible(lb2gt_5,857). happensAt(walking(lb2gt_5),857). orientation(lb2gt_6,151,857). coords(lb2gt_6,72,122,857).  visible(lb2gt_6,857). happensAt(inactive(lb2gt_6),857).
orientation(lb2gt_5,152,858). coords(lb2gt_5,85,94,858).  visible(lb2gt_5,858). happensAt(walking(lb2gt_5),858). orientation(lb2gt_6,151,858). coords(lb2gt_6,72,122,858).  visible(lb2gt_6,858). happensAt(inactive(lb2gt_6),858).
orientation(lb2gt_5,152,859). coords(lb2gt_5,85,96,859).  visible(lb2gt_5,859). happensAt(walking(lb2gt_5),859). orientation(lb2gt_6,151,859). coords(lb2gt_6,72,122,859).  visible(lb2gt_6,859). happensAt(inactive(lb2gt_6),859).
orientation(lb2gt_5,152,860). coords(lb2gt_5,85,98,860).  visible(lb2gt_5,860). happensAt(walking(lb2gt_5),860). orientation(lb2gt_6,151,860). coords(lb2gt_6,72,122,860).  visible(lb2gt_6,860). happensAt(inactive(lb2gt_6),860).
orientation(lb2gt_5,152,861). coords(lb2gt_5,84,99,861).  visible(lb2gt_5,861). happensAt(walking(lb2gt_5),861). orientation(lb2gt_6,151,861). coords(lb2gt_6,72,122,861).  visible(lb2gt_6,861). happensAt(inactive(lb2gt_6),861).
orientation(lb2gt_5,152,862). coords(lb2gt_5,84,100,862).  visible(lb2gt_5,862). happensAt(walking(lb2gt_5),862). orientation(lb2gt_6,151,862). coords(lb2gt_6,72,122,862).  visible(lb2gt_6,862). happensAt(inactive(lb2gt_6),862).
orientation(lb2gt_5,152,863). coords(lb2gt_5,83,100,863).  visible(lb2gt_5,863). happensAt(walking(lb2gt_5),863). orientation(lb2gt_6,151,863). coords(lb2gt_6,72,122,863).  visible(lb2gt_6,863). happensAt(inactive(lb2gt_6),863).
orientation(lb2gt_5,152,864). coords(lb2gt_5,82,101,864).  visible(lb2gt_5,864). happensAt(walking(lb2gt_5),864). orientation(lb2gt_6,151,864). coords(lb2gt_6,72,122,864).  visible(lb2gt_6,864). happensAt(inactive(lb2gt_6),864).
orientation(lb2gt_5,152,865). coords(lb2gt_5,81,101,865).  visible(lb2gt_5,865). happensAt(walking(lb2gt_5),865). orientation(lb2gt_6,151,865). coords(lb2gt_6,72,122,865).  visible(lb2gt_6,865). happensAt(inactive(lb2gt_6),865).
orientation(lb2gt_5,152,866). coords(lb2gt_5,81,102,866).  visible(lb2gt_5,866). happensAt(walking(lb2gt_5),866). orientation(lb2gt_6,151,866). coords(lb2gt_6,72,122,866).  visible(lb2gt_6,866). happensAt(inactive(lb2gt_6),866).
orientation(lb2gt_5,152,867). coords(lb2gt_5,80,102,867).  visible(lb2gt_5,867). happensAt(walking(lb2gt_5),867). orientation(lb2gt_6,151,867). coords(lb2gt_6,72,122,867).  visible(lb2gt_6,867). happensAt(inactive(lb2gt_6),867).
#show holdsAt/2.