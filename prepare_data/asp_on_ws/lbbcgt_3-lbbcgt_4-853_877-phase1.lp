
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
#show holdsAt/2.