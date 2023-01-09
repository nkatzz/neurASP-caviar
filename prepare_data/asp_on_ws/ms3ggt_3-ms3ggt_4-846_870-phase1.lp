
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

	orientation(ms3ggt_3,153,846). coords(ms3ggt_3,35,160,846).  visible(ms3ggt_3,846). happensAt(walking(ms3ggt_3),846). orientation(ms3ggt_4,140,846). coords(ms3ggt_4,135,38,846).  visible(ms3ggt_4,846). happensAt(walking(ms3ggt_4),846).
orientation(ms3ggt_3,153,847). coords(ms3ggt_3,35,160,847).  visible(ms3ggt_3,847). happensAt(walking(ms3ggt_3),847). orientation(ms3ggt_4,140,847). coords(ms3ggt_4,135,38,847).  visible(ms3ggt_4,847). happensAt(walking(ms3ggt_4),847).
orientation(ms3ggt_3,153,848). coords(ms3ggt_3,35,161,848).  visible(ms3ggt_3,848). happensAt(walking(ms3ggt_3),848). orientation(ms3ggt_4,140,848). coords(ms3ggt_4,135,38,848).  visible(ms3ggt_4,848). happensAt(walking(ms3ggt_4),848).
orientation(ms3ggt_3,153,849). coords(ms3ggt_3,35,163,849).  visible(ms3ggt_3,849). happensAt(walking(ms3ggt_3),849). orientation(ms3ggt_4,140,849). coords(ms3ggt_4,135,38,849).  visible(ms3ggt_4,849). happensAt(walking(ms3ggt_4),849).
orientation(ms3ggt_3,162,850). coords(ms3ggt_3,35,163,850).  visible(ms3ggt_3,850). happensAt(walking(ms3ggt_3),850). orientation(ms3ggt_4,140,850). coords(ms3ggt_4,135,39,850).  visible(ms3ggt_4,850). happensAt(walking(ms3ggt_4),850).
orientation(ms3ggt_3,162,851). coords(ms3ggt_3,35,166,851).  visible(ms3ggt_3,851). happensAt(walking(ms3ggt_3),851). orientation(ms3ggt_4,140,851). coords(ms3ggt_4,135,39,851).  visible(ms3ggt_4,851). happensAt(walking(ms3ggt_4),851).
orientation(ms3ggt_3,162,852). coords(ms3ggt_3,35,166,852).  visible(ms3ggt_3,852). happensAt(walking(ms3ggt_3),852). orientation(ms3ggt_4,140,852). coords(ms3ggt_4,135,39,852).  visible(ms3ggt_4,852). happensAt(walking(ms3ggt_4),852).
orientation(ms3ggt_3,162,853). coords(ms3ggt_3,35,167,853).  visible(ms3ggt_3,853). happensAt(walking(ms3ggt_3),853). orientation(ms3ggt_4,140,853). coords(ms3ggt_4,135,39,853).  visible(ms3ggt_4,853). happensAt(walking(ms3ggt_4),853).
orientation(ms3ggt_3,162,854). coords(ms3ggt_3,34,170,854).  visible(ms3ggt_3,854). happensAt(walking(ms3ggt_3),854). orientation(ms3ggt_4,140,854). coords(ms3ggt_4,134,39,854).  visible(ms3ggt_4,854). happensAt(walking(ms3ggt_4),854).
orientation(ms3ggt_3,162,855). coords(ms3ggt_3,34,171,855).  visible(ms3ggt_3,855). happensAt(walking(ms3ggt_3),855). orientation(ms3ggt_4,140,855). coords(ms3ggt_4,134,39,855).  visible(ms3ggt_4,855). happensAt(walking(ms3ggt_4),855).
orientation(ms3ggt_3,162,856). coords(ms3ggt_3,34,171,856).  visible(ms3ggt_3,856). happensAt(walking(ms3ggt_3),856). orientation(ms3ggt_4,140,856). coords(ms3ggt_4,134,39,856).  visible(ms3ggt_4,856). happensAt(walking(ms3ggt_4),856).
orientation(ms3ggt_3,162,857). coords(ms3ggt_3,34,172,857).  visible(ms3ggt_3,857). happensAt(walking(ms3ggt_3),857). orientation(ms3ggt_4,140,857). coords(ms3ggt_4,134,39,857).  visible(ms3ggt_4,857). happensAt(walking(ms3ggt_4),857).
orientation(ms3ggt_3,162,858). coords(ms3ggt_3,33,172,858).  visible(ms3ggt_3,858). happensAt(walking(ms3ggt_3),858). orientation(ms3ggt_4,140,858). coords(ms3ggt_4,133,40,858).  visible(ms3ggt_4,858). happensAt(walking(ms3ggt_4),858).
orientation(ms3ggt_3,162,859). coords(ms3ggt_3,33,173,859).  visible(ms3ggt_3,859). happensAt(walking(ms3ggt_3),859). orientation(ms3ggt_4,140,859). coords(ms3ggt_4,133,40,859).  visible(ms3ggt_4,859). happensAt(walking(ms3ggt_4),859).
orientation(ms3ggt_3,162,860). coords(ms3ggt_3,33,173,860).  visible(ms3ggt_3,860). happensAt(walking(ms3ggt_3),860). orientation(ms3ggt_4,140,860). coords(ms3ggt_4,133,40,860).  visible(ms3ggt_4,860). happensAt(walking(ms3ggt_4),860).
orientation(ms3ggt_3,162,861). coords(ms3ggt_3,33,175,861).  visible(ms3ggt_3,861). happensAt(walking(ms3ggt_3),861). orientation(ms3ggt_4,140,861). coords(ms3ggt_4,133,40,861).  visible(ms3ggt_4,861). happensAt(walking(ms3ggt_4),861).
orientation(ms3ggt_3,162,862). coords(ms3ggt_3,33,175,862).  visible(ms3ggt_3,862). happensAt(walking(ms3ggt_3),862). orientation(ms3ggt_4,140,862). coords(ms3ggt_4,132,41,862).  visible(ms3ggt_4,862). happensAt(walking(ms3ggt_4),862).
orientation(ms3ggt_3,162,863). coords(ms3ggt_3,33,177,863).  visible(ms3ggt_3,863). happensAt(walking(ms3ggt_3),863). orientation(ms3ggt_4,140,863). coords(ms3ggt_4,132,41,863).  visible(ms3ggt_4,863). happensAt(walking(ms3ggt_4),863).
orientation(ms3ggt_3,162,864). coords(ms3ggt_3,33,179,864).  visible(ms3ggt_3,864). happensAt(walking(ms3ggt_3),864). orientation(ms3ggt_4,140,864). coords(ms3ggt_4,132,41,864).  visible(ms3ggt_4,864). happensAt(walking(ms3ggt_4),864).
orientation(ms3ggt_3,162,865). coords(ms3ggt_3,33,179,865).  visible(ms3ggt_3,865). happensAt(walking(ms3ggt_3),865). orientation(ms3ggt_4,140,865). coords(ms3ggt_4,132,41,865).  visible(ms3ggt_4,865). happensAt(walking(ms3ggt_4),865).
orientation(ms3ggt_3,162,866). coords(ms3ggt_3,33,180,866).  visible(ms3ggt_3,866). happensAt(walking(ms3ggt_3),866). orientation(ms3ggt_4,140,866). coords(ms3ggt_4,132,41,866).  visible(ms3ggt_4,866). happensAt(walking(ms3ggt_4),866).
orientation(ms3ggt_3,162,867). coords(ms3ggt_3,33,181,867).  visible(ms3ggt_3,867). happensAt(walking(ms3ggt_3),867). orientation(ms3ggt_4,140,867). coords(ms3ggt_4,132,41,867).  visible(ms3ggt_4,867). happensAt(walking(ms3ggt_4),867).
orientation(ms3ggt_3,162,868). coords(ms3ggt_3,33,182,868).  visible(ms3ggt_3,868). happensAt(walking(ms3ggt_3),868). orientation(ms3ggt_4,140,868). coords(ms3ggt_4,132,41,868).  visible(ms3ggt_4,868). happensAt(walking(ms3ggt_4),868).
orientation(ms3ggt_3,162,869). coords(ms3ggt_3,33,183,869).  visible(ms3ggt_3,869). happensAt(walking(ms3ggt_3),869). orientation(ms3ggt_4,140,869). coords(ms3ggt_4,131,42,869).  visible(ms3ggt_4,869). happensAt(walking(ms3ggt_4),869).
orientation(ms3ggt_3,162,870). coords(ms3ggt_3,33,184,870).  visible(ms3ggt_3,870). happensAt(walking(ms3ggt_3),870). orientation(ms3ggt_4,140,870). coords(ms3ggt_4,131,42,870).  visible(ms3ggt_4,870). happensAt(walking(ms3ggt_4),870).
#show holdsAt/2.