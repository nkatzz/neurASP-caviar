
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

	orientation(lbbcgt_3,150,841). coords(lbbcgt_3,44,113,841).  visible(lbbcgt_3,841). happensAt(walking(lbbcgt_3),841). orientation(lbbcgt_4,131,841). coords(lbbcgt_4,3,95,841).  happensAt(appear(lbbcgt_4),841). happensAt(walking(lbbcgt_4),841).
orientation(lbbcgt_3,150,842). coords(lbbcgt_3,44,114,842).  visible(lbbcgt_3,842). happensAt(walking(lbbcgt_3),842). orientation(lbbcgt_4,131,842). coords(lbbcgt_4,3,95,842).  visible(lbbcgt_4,842). happensAt(walking(lbbcgt_4),842).
orientation(lbbcgt_3,150,843). coords(lbbcgt_3,44,114,843).  visible(lbbcgt_3,843). happensAt(walking(lbbcgt_3),843). orientation(lbbcgt_4,131,843). coords(lbbcgt_4,3,95,843).  visible(lbbcgt_4,843). happensAt(walking(lbbcgt_4),843).
orientation(lbbcgt_3,150,844). coords(lbbcgt_3,44,115,844).  visible(lbbcgt_3,844). happensAt(walking(lbbcgt_3),844). orientation(lbbcgt_4,131,844). coords(lbbcgt_4,3,94,844).  visible(lbbcgt_4,844). happensAt(walking(lbbcgt_4),844).
orientation(lbbcgt_3,150,845). coords(lbbcgt_3,44,115,845).  visible(lbbcgt_3,845). happensAt(walking(lbbcgt_3),845). orientation(lbbcgt_4,131,845). coords(lbbcgt_4,3,94,845).  visible(lbbcgt_4,845). happensAt(walking(lbbcgt_4),845).
orientation(lbbcgt_3,150,846). coords(lbbcgt_3,44,116,846).  visible(lbbcgt_3,846). happensAt(walking(lbbcgt_3),846). orientation(lbbcgt_4,131,846). coords(lbbcgt_4,3,94,846).  visible(lbbcgt_4,846). happensAt(walking(lbbcgt_4),846).
orientation(lbbcgt_3,150,847). coords(lbbcgt_3,44,116,847).  visible(lbbcgt_3,847). happensAt(walking(lbbcgt_3),847). orientation(lbbcgt_4,131,847). coords(lbbcgt_4,3,92,847).  visible(lbbcgt_4,847). happensAt(walking(lbbcgt_4),847).
orientation(lbbcgt_3,150,848). coords(lbbcgt_3,44,117,848).  visible(lbbcgt_3,848). happensAt(walking(lbbcgt_3),848). orientation(lbbcgt_4,131,848). coords(lbbcgt_4,3,91,848).  visible(lbbcgt_4,848). happensAt(walking(lbbcgt_4),848).
orientation(lbbcgt_3,150,849). coords(lbbcgt_3,45,117,849).  visible(lbbcgt_3,849). happensAt(walking(lbbcgt_3),849). orientation(lbbcgt_4,131,849). coords(lbbcgt_4,3,89,849).  visible(lbbcgt_4,849). happensAt(walking(lbbcgt_4),849).
orientation(lbbcgt_3,150,850). coords(lbbcgt_3,44,119,850).  visible(lbbcgt_3,850). happensAt(walking(lbbcgt_3),850). orientation(lbbcgt_4,131,850). coords(lbbcgt_4,3,89,850).  visible(lbbcgt_4,850). happensAt(walking(lbbcgt_4),850).
orientation(lbbcgt_3,150,851). coords(lbbcgt_3,45,120,851).  visible(lbbcgt_3,851). happensAt(walking(lbbcgt_3),851). orientation(lbbcgt_4,131,851). coords(lbbcgt_4,4,89,851).  visible(lbbcgt_4,851). happensAt(walking(lbbcgt_4),851).
orientation(lbbcgt_3,150,852). coords(lbbcgt_3,45,121,852).  visible(lbbcgt_3,852). happensAt(walking(lbbcgt_3),852). orientation(lbbcgt_4,131,852). coords(lbbcgt_4,4,87,852).  visible(lbbcgt_4,852). happensAt(walking(lbbcgt_4),852).
orientation(lbbcgt_3,150,853). coords(lbbcgt_3,45,121,853).  visible(lbbcgt_3,853). happensAt(walking(lbbcgt_3),853). orientation(lbbcgt_4,131,853). coords(lbbcgt_4,4,87,853).  visible(lbbcgt_4,853). happensAt(walking(lbbcgt_4),853).
orientation(lbbcgt_3,150,854). coords(lbbcgt_3,45,121,854).  visible(lbbcgt_3,854). happensAt(walking(lbbcgt_3),854). orientation(lbbcgt_4,131,854). coords(lbbcgt_4,5,86,854).  visible(lbbcgt_4,854). happensAt(walking(lbbcgt_4),854).
orientation(lbbcgt_3,150,855). coords(lbbcgt_3,44,121,855).  visible(lbbcgt_3,855). happensAt(walking(lbbcgt_3),855). orientation(lbbcgt_4,131,855). coords(lbbcgt_4,5,85,855).  visible(lbbcgt_4,855). happensAt(walking(lbbcgt_4),855).
orientation(lbbcgt_3,150,856). coords(lbbcgt_3,44,122,856).  visible(lbbcgt_3,856). happensAt(walking(lbbcgt_3),856). orientation(lbbcgt_4,131,856). coords(lbbcgt_4,5,84,856).  visible(lbbcgt_4,856). happensAt(walking(lbbcgt_4),856).
orientation(lbbcgt_3,150,857). coords(lbbcgt_3,44,122,857).  visible(lbbcgt_3,857). happensAt(walking(lbbcgt_3),857). orientation(lbbcgt_4,131,857). coords(lbbcgt_4,5,83,857).  visible(lbbcgt_4,857). happensAt(walking(lbbcgt_4),857).
orientation(lbbcgt_3,158,858). coords(lbbcgt_3,44,122,858).  visible(lbbcgt_3,858). happensAt(walking(lbbcgt_3),858). orientation(lbbcgt_4,131,858). coords(lbbcgt_4,6,81,858).  visible(lbbcgt_4,858). happensAt(walking(lbbcgt_4),858).
orientation(lbbcgt_3,158,859). coords(lbbcgt_3,44,123,859).  visible(lbbcgt_3,859). happensAt(walking(lbbcgt_3),859). orientation(lbbcgt_4,131,859). coords(lbbcgt_4,6,79,859).  visible(lbbcgt_4,859). happensAt(walking(lbbcgt_4),859).
orientation(lbbcgt_3,158,860). coords(lbbcgt_3,44,123,860).  visible(lbbcgt_3,860). happensAt(walking(lbbcgt_3),860). orientation(lbbcgt_4,131,860). coords(lbbcgt_4,6,79,860).  visible(lbbcgt_4,860). happensAt(walking(lbbcgt_4),860).
orientation(lbbcgt_3,158,861). coords(lbbcgt_3,44,124,861).  visible(lbbcgt_3,861). happensAt(walking(lbbcgt_3),861). orientation(lbbcgt_4,131,861). coords(lbbcgt_4,6,80,861).  visible(lbbcgt_4,861). happensAt(walking(lbbcgt_4),861).
orientation(lbbcgt_3,158,862). coords(lbbcgt_3,44,125,862).  visible(lbbcgt_3,862). happensAt(walking(lbbcgt_3),862). orientation(lbbcgt_4,131,862). coords(lbbcgt_4,6,79,862).  visible(lbbcgt_4,862). happensAt(walking(lbbcgt_4),862).
orientation(lbbcgt_3,158,863). coords(lbbcgt_3,43,127,863).  visible(lbbcgt_3,863). happensAt(walking(lbbcgt_3),863). orientation(lbbcgt_4,131,863). coords(lbbcgt_4,6,78,863).  visible(lbbcgt_4,863). happensAt(walking(lbbcgt_4),863).
orientation(lbbcgt_3,158,864). coords(lbbcgt_3,44,128,864).  visible(lbbcgt_3,864). happensAt(walking(lbbcgt_3),864). orientation(lbbcgt_4,137,864). coords(lbbcgt_4,7,77,864).  visible(lbbcgt_4,864). happensAt(walking(lbbcgt_4),864).
orientation(lbbcgt_3,158,865). coords(lbbcgt_3,44,129,865).  visible(lbbcgt_3,865). happensAt(walking(lbbcgt_3),865). orientation(lbbcgt_4,137,865). coords(lbbcgt_4,8,76,865).  visible(lbbcgt_4,865). happensAt(walking(lbbcgt_4),865).
#show holdsAt/2.