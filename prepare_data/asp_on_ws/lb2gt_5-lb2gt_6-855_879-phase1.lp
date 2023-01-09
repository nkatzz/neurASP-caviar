
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
orientation(lb2gt_5,152,868). coords(lb2gt_5,78,102,868).  visible(lb2gt_5,868). happensAt(walking(lb2gt_5),868). orientation(lb2gt_6,151,868). coords(lb2gt_6,72,122,868).  visible(lb2gt_6,868). happensAt(inactive(lb2gt_6),868).
orientation(lb2gt_5,152,869). coords(lb2gt_5,76,102,869).  visible(lb2gt_5,869). happensAt(walking(lb2gt_5),869). orientation(lb2gt_6,151,869). coords(lb2gt_6,72,122,869).  visible(lb2gt_6,869). happensAt(inactive(lb2gt_6),869).
orientation(lb2gt_5,152,870). coords(lb2gt_5,76,103,870).  visible(lb2gt_5,870). happensAt(walking(lb2gt_5),870). orientation(lb2gt_6,151,870). coords(lb2gt_6,72,122,870).  visible(lb2gt_6,870). happensAt(inactive(lb2gt_6),870).
orientation(lb2gt_5,152,871). coords(lb2gt_5,75,104,871).  visible(lb2gt_5,871). happensAt(walking(lb2gt_5),871). orientation(lb2gt_6,151,871). coords(lb2gt_6,72,122,871).  visible(lb2gt_6,871). happensAt(inactive(lb2gt_6),871).
orientation(lb2gt_5,152,872). coords(lb2gt_5,75,105,872).  visible(lb2gt_5,872). happensAt(walking(lb2gt_5),872). orientation(lb2gt_6,151,872). coords(lb2gt_6,72,122,872).  visible(lb2gt_6,872). happensAt(inactive(lb2gt_6),872).
orientation(lb2gt_5,152,873). coords(lb2gt_5,75,106,873).  visible(lb2gt_5,873). happensAt(walking(lb2gt_5),873). orientation(lb2gt_6,151,873). coords(lb2gt_6,72,122,873).  visible(lb2gt_6,873). happensAt(inactive(lb2gt_6),873).
orientation(lb2gt_5,152,874). coords(lb2gt_5,75,108,874).  visible(lb2gt_5,874). happensAt(walking(lb2gt_5),874). orientation(lb2gt_6,151,874). coords(lb2gt_6,72,122,874).  visible(lb2gt_6,874). happensAt(inactive(lb2gt_6),874).
orientation(lb2gt_5,152,875). coords(lb2gt_5,75,108,875).  visible(lb2gt_5,875). happensAt(walking(lb2gt_5),875). orientation(lb2gt_6,151,875). coords(lb2gt_6,72,122,875).  visible(lb2gt_6,875). happensAt(inactive(lb2gt_6),875).
orientation(lb2gt_5,152,876). coords(lb2gt_5,75,109,876).  visible(lb2gt_5,876). happensAt(walking(lb2gt_5),876). orientation(lb2gt_6,151,876). coords(lb2gt_6,72,122,876).  visible(lb2gt_6,876). happensAt(inactive(lb2gt_6),876).
orientation(lb2gt_5,152,877). coords(lb2gt_5,74,110,877).  visible(lb2gt_5,877). happensAt(walking(lb2gt_5),877). orientation(lb2gt_6,151,877). coords(lb2gt_6,72,122,877).  visible(lb2gt_6,877). happensAt(inactive(lb2gt_6),877).
orientation(lb2gt_5,152,878). coords(lb2gt_5,74,111,878).  visible(lb2gt_5,878). happensAt(walking(lb2gt_5),878). orientation(lb2gt_6,151,878). coords(lb2gt_6,72,122,878).  visible(lb2gt_6,878). happensAt(inactive(lb2gt_6),878).
orientation(lb2gt_5,152,879). coords(lb2gt_5,73,111,879).  visible(lb2gt_5,879). happensAt(walking(lb2gt_5),879). orientation(lb2gt_6,151,879). coords(lb2gt_6,72,122,879).  visible(lb2gt_6,879). happensAt(inactive(lb2gt_6),879).
#show holdsAt/2.