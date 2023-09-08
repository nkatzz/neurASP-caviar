
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

	orientation(lbpugt_2,84,851). coords(lbpugt_2,292,95,851).  visible(lbpugt_2,851). happensAt(active(lbpugt_2),851). orientation(lbpugt_4,135,851). coords(lbpugt_4,219,34,851).  visible(lbpugt_4,851). happensAt(inactive(lbpugt_4),851).
orientation(lbpugt_2,84,852). coords(lbpugt_2,292,95,852).  visible(lbpugt_2,852). happensAt(active(lbpugt_2),852). orientation(lbpugt_4,135,852). coords(lbpugt_4,219,34,852).  visible(lbpugt_4,852). happensAt(inactive(lbpugt_4),852).
orientation(lbpugt_2,84,853). coords(lbpugt_2,292,95,853).  visible(lbpugt_2,853). happensAt(active(lbpugt_2),853). orientation(lbpugt_4,135,853). coords(lbpugt_4,219,34,853).  visible(lbpugt_4,853). happensAt(inactive(lbpugt_4),853).
orientation(lbpugt_2,84,854). coords(lbpugt_2,292,96,854).  visible(lbpugt_2,854). happensAt(active(lbpugt_2),854). orientation(lbpugt_4,135,854). coords(lbpugt_4,219,34,854).  visible(lbpugt_4,854). happensAt(inactive(lbpugt_4),854).
orientation(lbpugt_2,84,855). coords(lbpugt_2,292,96,855).  visible(lbpugt_2,855). happensAt(active(lbpugt_2),855). orientation(lbpugt_4,135,855). coords(lbpugt_4,219,34,855).  visible(lbpugt_4,855). happensAt(inactive(lbpugt_4),855).
orientation(lbpugt_2,84,856). coords(lbpugt_2,292,96,856).  visible(lbpugt_2,856). happensAt(active(lbpugt_2),856). orientation(lbpugt_4,135,856). coords(lbpugt_4,219,34,856).  visible(lbpugt_4,856). happensAt(inactive(lbpugt_4),856).
orientation(lbpugt_2,84,857). coords(lbpugt_2,292,96,857).  visible(lbpugt_2,857). happensAt(active(lbpugt_2),857). orientation(lbpugt_4,135,857). coords(lbpugt_4,219,34,857).  visible(lbpugt_4,857). happensAt(inactive(lbpugt_4),857).
orientation(lbpugt_2,84,858). coords(lbpugt_2,292,97,858).  visible(lbpugt_2,858). happensAt(active(lbpugt_2),858). orientation(lbpugt_4,135,858). coords(lbpugt_4,219,34,858).  visible(lbpugt_4,858). happensAt(inactive(lbpugt_4),858).
orientation(lbpugt_2,84,859). coords(lbpugt_2,292,98,859).  visible(lbpugt_2,859). happensAt(walking(lbpugt_2),859). orientation(lbpugt_4,135,859). coords(lbpugt_4,219,34,859).  visible(lbpugt_4,859). happensAt(inactive(lbpugt_4),859).
orientation(lbpugt_2,84,860). coords(lbpugt_2,291,98,860).  visible(lbpugt_2,860). happensAt(walking(lbpugt_2),860). orientation(lbpugt_4,135,860). coords(lbpugt_4,219,34,860).  visible(lbpugt_4,860). happensAt(inactive(lbpugt_4),860).
orientation(lbpugt_2,84,861). coords(lbpugt_2,290,99,861).  visible(lbpugt_2,861). happensAt(walking(lbpugt_2),861). orientation(lbpugt_4,135,861). coords(lbpugt_4,219,34,861).  visible(lbpugt_4,861). happensAt(inactive(lbpugt_4),861).
orientation(lbpugt_2,84,862). coords(lbpugt_2,290,100,862).  visible(lbpugt_2,862). happensAt(walking(lbpugt_2),862). orientation(lbpugt_4,135,862). coords(lbpugt_4,219,34,862).  visible(lbpugt_4,862). happensAt(inactive(lbpugt_4),862).
orientation(lbpugt_2,87,863). coords(lbpugt_2,290,101,863).  visible(lbpugt_2,863). happensAt(walking(lbpugt_2),863). orientation(lbpugt_4,135,863). coords(lbpugt_4,219,34,863).  visible(lbpugt_4,863). happensAt(inactive(lbpugt_4),863).
orientation(lbpugt_2,89,864). coords(lbpugt_2,290,101,864).  visible(lbpugt_2,864). happensAt(walking(lbpugt_2),864). orientation(lbpugt_4,135,864). coords(lbpugt_4,219,34,864).  visible(lbpugt_4,864). happensAt(inactive(lbpugt_4),864).
orientation(lbpugt_2,92,865). coords(lbpugt_2,289,102,865).  visible(lbpugt_2,865). happensAt(walking(lbpugt_2),865). orientation(lbpugt_4,135,865). coords(lbpugt_4,219,34,865).  visible(lbpugt_4,865). happensAt(inactive(lbpugt_4),865).
orientation(lbpugt_2,93,866). coords(lbpugt_2,289,103,866).  visible(lbpugt_2,866). happensAt(walking(lbpugt_2),866). orientation(lbpugt_4,135,866). coords(lbpugt_4,219,34,866).  visible(lbpugt_4,866). happensAt(inactive(lbpugt_4),866).
orientation(lbpugt_2,100,867). coords(lbpugt_2,289,103,867).  visible(lbpugt_2,867). happensAt(walking(lbpugt_2),867). orientation(lbpugt_4,135,867). coords(lbpugt_4,219,34,867).  visible(lbpugt_4,867). happensAt(inactive(lbpugt_4),867).
orientation(lbpugt_2,100,868). coords(lbpugt_2,288,104,868).  visible(lbpugt_2,868). happensAt(walking(lbpugt_2),868). orientation(lbpugt_4,135,868). coords(lbpugt_4,219,34,868).  visible(lbpugt_4,868). happensAt(inactive(lbpugt_4),868).
orientation(lbpugt_2,109,869). coords(lbpugt_2,288,105,869).  visible(lbpugt_2,869). happensAt(walking(lbpugt_2),869). orientation(lbpugt_4,135,869). coords(lbpugt_4,219,34,869).  visible(lbpugt_4,869). happensAt(inactive(lbpugt_4),869).
orientation(lbpugt_2,109,870). coords(lbpugt_2,287,105,870).  visible(lbpugt_2,870). happensAt(walking(lbpugt_2),870). orientation(lbpugt_4,135,870). coords(lbpugt_4,219,34,870).  visible(lbpugt_4,870). happensAt(inactive(lbpugt_4),870).
orientation(lbpugt_2,116,871). coords(lbpugt_2,286,107,871).  visible(lbpugt_2,871). happensAt(walking(lbpugt_2),871). orientation(lbpugt_4,135,871). coords(lbpugt_4,219,34,871).  visible(lbpugt_4,871). happensAt(inactive(lbpugt_4),871).
orientation(lbpugt_2,116,872). coords(lbpugt_2,286,107,872).  visible(lbpugt_2,872). happensAt(walking(lbpugt_2),872). orientation(lbpugt_4,135,872). coords(lbpugt_4,219,34,872).  visible(lbpugt_4,872). happensAt(inactive(lbpugt_4),872).
orientation(lbpugt_2,123,873). coords(lbpugt_2,285,108,873).  visible(lbpugt_2,873). happensAt(walking(lbpugt_2),873). orientation(lbpugt_4,135,873). coords(lbpugt_4,219,34,873).  visible(lbpugt_4,873). happensAt(inactive(lbpugt_4),873).
orientation(lbpugt_2,123,874). coords(lbpugt_2,285,108,874).  visible(lbpugt_2,874). happensAt(walking(lbpugt_2),874). orientation(lbpugt_4,135,874). coords(lbpugt_4,219,34,874).  visible(lbpugt_4,874). happensAt(inactive(lbpugt_4),874).
orientation(lbpugt_2,128,875). coords(lbpugt_2,284,109,875).  visible(lbpugt_2,875). happensAt(walking(lbpugt_2),875). orientation(lbpugt_4,135,875). coords(lbpugt_4,219,34,875).  visible(lbpugt_4,875). happensAt(inactive(lbpugt_4),875).
#show holdsAt/2.