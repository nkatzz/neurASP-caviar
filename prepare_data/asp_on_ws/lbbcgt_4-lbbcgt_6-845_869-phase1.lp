
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

	orientation(lbbcgt_4,131,845). coords(lbbcgt_4,3,94,845).  visible(lbbcgt_4,845). happensAt(walking(lbbcgt_4),845). orientation(lbbcgt_6,118,845). coords(lbbcgt_6,172,21,845).  happensAt(appear(lbbcgt_6),845). happensAt(running(lbbcgt_6),845).
orientation(lbbcgt_4,131,846). coords(lbbcgt_4,3,94,846).  visible(lbbcgt_4,846). happensAt(walking(lbbcgt_4),846). orientation(lbbcgt_6,118,846). coords(lbbcgt_6,172,21,846).  visible(lbbcgt_6,846). happensAt(running(lbbcgt_6),846).
orientation(lbbcgt_4,131,847). coords(lbbcgt_4,3,92,847).  visible(lbbcgt_4,847). happensAt(walking(lbbcgt_4),847). orientation(lbbcgt_6,118,847). coords(lbbcgt_6,172,21,847).  visible(lbbcgt_6,847). happensAt(running(lbbcgt_6),847).
orientation(lbbcgt_4,131,848). coords(lbbcgt_4,3,91,848).  visible(lbbcgt_4,848). happensAt(walking(lbbcgt_4),848). orientation(lbbcgt_6,118,848). coords(lbbcgt_6,172,23,848).  visible(lbbcgt_6,848). happensAt(running(lbbcgt_6),848).
orientation(lbbcgt_4,131,849). coords(lbbcgt_4,3,89,849).  visible(lbbcgt_4,849). happensAt(walking(lbbcgt_4),849). orientation(lbbcgt_6,118,849). coords(lbbcgt_6,172,25,849).  visible(lbbcgt_6,849). happensAt(running(lbbcgt_6),849).
orientation(lbbcgt_4,131,850). coords(lbbcgt_4,3,89,850).  visible(lbbcgt_4,850). happensAt(walking(lbbcgt_4),850). orientation(lbbcgt_6,118,850). coords(lbbcgt_6,172,25,850).  visible(lbbcgt_6,850). happensAt(running(lbbcgt_6),850).
orientation(lbbcgt_4,131,851). coords(lbbcgt_4,4,89,851).  visible(lbbcgt_4,851). happensAt(walking(lbbcgt_4),851). orientation(lbbcgt_6,118,851). coords(lbbcgt_6,172,25,851).  visible(lbbcgt_6,851). happensAt(running(lbbcgt_6),851).
orientation(lbbcgt_4,131,852). coords(lbbcgt_4,4,87,852).  visible(lbbcgt_4,852). happensAt(walking(lbbcgt_4),852). orientation(lbbcgt_6,118,852). coords(lbbcgt_6,171,25,852).  visible(lbbcgt_6,852). happensAt(running(lbbcgt_6),852).
orientation(lbbcgt_4,131,853). coords(lbbcgt_4,4,87,853).  visible(lbbcgt_4,853). happensAt(walking(lbbcgt_4),853). orientation(lbbcgt_6,123,853). coords(lbbcgt_6,173,31,853).  visible(lbbcgt_6,853). happensAt(running(lbbcgt_6),853).
orientation(lbbcgt_4,131,854). coords(lbbcgt_4,5,86,854).  visible(lbbcgt_4,854). happensAt(walking(lbbcgt_4),854). orientation(lbbcgt_6,127,854). coords(lbbcgt_6,173,32,854).  visible(lbbcgt_6,854). happensAt(running(lbbcgt_6),854).
orientation(lbbcgt_4,131,855). coords(lbbcgt_4,5,85,855).  visible(lbbcgt_4,855). happensAt(walking(lbbcgt_4),855). orientation(lbbcgt_6,135,855). coords(lbbcgt_6,172,32,855).  visible(lbbcgt_6,855). happensAt(running(lbbcgt_6),855).
orientation(lbbcgt_4,131,856). coords(lbbcgt_4,5,84,856).  visible(lbbcgt_4,856). happensAt(walking(lbbcgt_4),856). orientation(lbbcgt_6,133,856). coords(lbbcgt_6,172,32,856).  visible(lbbcgt_6,856). happensAt(running(lbbcgt_6),856).
orientation(lbbcgt_4,131,857). coords(lbbcgt_4,5,83,857).  visible(lbbcgt_4,857). happensAt(walking(lbbcgt_4),857). orientation(lbbcgt_6,130,857). coords(lbbcgt_6,171,31,857).  visible(lbbcgt_6,857). happensAt(running(lbbcgt_6),857).
orientation(lbbcgt_4,131,858). coords(lbbcgt_4,6,81,858).  visible(lbbcgt_4,858). happensAt(walking(lbbcgt_4),858). orientation(lbbcgt_6,130,858). coords(lbbcgt_6,170,32,858).  visible(lbbcgt_6,858). happensAt(running(lbbcgt_6),858).
orientation(lbbcgt_4,131,859). coords(lbbcgt_4,6,79,859).  visible(lbbcgt_4,859). happensAt(walking(lbbcgt_4),859). orientation(lbbcgt_6,130,859). coords(lbbcgt_6,169,32,859).  visible(lbbcgt_6,859). happensAt(running(lbbcgt_6),859).
orientation(lbbcgt_4,131,860). coords(lbbcgt_4,6,79,860).  visible(lbbcgt_4,860). happensAt(walking(lbbcgt_4),860). orientation(lbbcgt_6,130,860). coords(lbbcgt_6,168,33,860).  visible(lbbcgt_6,860). happensAt(running(lbbcgt_6),860).
orientation(lbbcgt_4,131,861). coords(lbbcgt_4,6,80,861).  visible(lbbcgt_4,861). happensAt(walking(lbbcgt_4),861). orientation(lbbcgt_6,130,861). coords(lbbcgt_6,167,35,861).  visible(lbbcgt_6,861). happensAt(running(lbbcgt_6),861).
orientation(lbbcgt_4,131,862). coords(lbbcgt_4,6,79,862).  visible(lbbcgt_4,862). happensAt(walking(lbbcgt_4),862). orientation(lbbcgt_6,130,862). coords(lbbcgt_6,167,36,862).  visible(lbbcgt_6,862). happensAt(running(lbbcgt_6),862).
orientation(lbbcgt_4,131,863). coords(lbbcgt_4,6,78,863).  visible(lbbcgt_4,863). happensAt(walking(lbbcgt_4),863). orientation(lbbcgt_6,130,863). coords(lbbcgt_6,166,37,863).  visible(lbbcgt_6,863). happensAt(running(lbbcgt_6),863).
orientation(lbbcgt_4,137,864). coords(lbbcgt_4,7,77,864).  visible(lbbcgt_4,864). happensAt(walking(lbbcgt_4),864). orientation(lbbcgt_6,130,864). coords(lbbcgt_6,166,37,864).  visible(lbbcgt_6,864). happensAt(running(lbbcgt_6),864).
orientation(lbbcgt_4,137,865). coords(lbbcgt_4,8,76,865).  visible(lbbcgt_4,865). happensAt(walking(lbbcgt_4),865). orientation(lbbcgt_6,130,865). coords(lbbcgt_6,165,37,865).  visible(lbbcgt_6,865). happensAt(running(lbbcgt_6),865).
orientation(lbbcgt_4,137,866). coords(lbbcgt_4,7,75,866).  visible(lbbcgt_4,866). happensAt(walking(lbbcgt_4),866). orientation(lbbcgt_6,130,866). coords(lbbcgt_6,164,38,866).  visible(lbbcgt_6,866). happensAt(running(lbbcgt_6),866).
orientation(lbbcgt_4,142,867). coords(lbbcgt_4,8,75,867).  visible(lbbcgt_4,867). happensAt(walking(lbbcgt_4),867). orientation(lbbcgt_6,130,867). coords(lbbcgt_6,162,38,867).  visible(lbbcgt_6,867). happensAt(running(lbbcgt_6),867).
orientation(lbbcgt_4,142,868). coords(lbbcgt_4,9,74,868).  visible(lbbcgt_4,868). happensAt(walking(lbbcgt_4),868). orientation(lbbcgt_6,130,868). coords(lbbcgt_6,161,38,868).  visible(lbbcgt_6,868). happensAt(running(lbbcgt_6),868).
orientation(lbbcgt_4,142,869). coords(lbbcgt_4,9,74,869).  visible(lbbcgt_4,869). happensAt(walking(lbbcgt_4),869). orientation(lbbcgt_6,130,869). coords(lbbcgt_6,160,38,869).  visible(lbbcgt_6,869). happensAt(running(lbbcgt_6),869).
#show holdsAt/2.