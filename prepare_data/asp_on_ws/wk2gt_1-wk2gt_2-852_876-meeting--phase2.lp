

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,127,852). coords(wk2gt_1,115,53,852).  visible(wk2gt_1,852). orientation(wk2gt_2,142,852). coords(wk2gt_2,99,63,852).  visible(wk2gt_2,852).
orientation(wk2gt_1,127,853). coords(wk2gt_1,115,53,853).  visible(wk2gt_1,853). orientation(wk2gt_2,142,853). coords(wk2gt_2,99,63,853).  visible(wk2gt_2,853).
orientation(wk2gt_1,127,854). coords(wk2gt_1,115,53,854).  visible(wk2gt_1,854). orientation(wk2gt_2,142,854). coords(wk2gt_2,99,63,854).  visible(wk2gt_2,854).
orientation(wk2gt_1,127,855). coords(wk2gt_1,115,53,855).  visible(wk2gt_1,855). orientation(wk2gt_2,142,855). coords(wk2gt_2,99,63,855).  visible(wk2gt_2,855).
orientation(wk2gt_1,127,856). coords(wk2gt_1,115,53,856).  visible(wk2gt_1,856). orientation(wk2gt_2,142,856). coords(wk2gt_2,99,63,856).  visible(wk2gt_2,856).
orientation(wk2gt_1,127,857). coords(wk2gt_1,115,53,857).  visible(wk2gt_1,857). orientation(wk2gt_2,142,857). coords(wk2gt_2,99,63,857).  visible(wk2gt_2,857).
orientation(wk2gt_1,127,858). coords(wk2gt_1,115,53,858).  visible(wk2gt_1,858). orientation(wk2gt_2,142,858). coords(wk2gt_2,99,63,858).  visible(wk2gt_2,858).
orientation(wk2gt_1,127,859). coords(wk2gt_1,115,53,859).  visible(wk2gt_1,859). orientation(wk2gt_2,142,859). coords(wk2gt_2,99,63,859).  visible(wk2gt_2,859).
orientation(wk2gt_1,127,860). coords(wk2gt_1,115,53,860).  visible(wk2gt_1,860). orientation(wk2gt_2,142,860). coords(wk2gt_2,99,63,860).  visible(wk2gt_2,860).
orientation(wk2gt_1,127,861). coords(wk2gt_1,115,53,861).  visible(wk2gt_1,861). orientation(wk2gt_2,142,861). coords(wk2gt_2,99,63,861).  visible(wk2gt_2,861).
orientation(wk2gt_1,127,862). coords(wk2gt_1,115,53,862).  visible(wk2gt_1,862). orientation(wk2gt_2,142,862). coords(wk2gt_2,99,63,862).  visible(wk2gt_2,862).
orientation(wk2gt_1,127,863). coords(wk2gt_1,115,53,863).  visible(wk2gt_1,863). orientation(wk2gt_2,142,863). coords(wk2gt_2,99,63,863).  visible(wk2gt_2,863).
orientation(wk2gt_1,127,864). coords(wk2gt_1,115,53,864).  visible(wk2gt_1,864). orientation(wk2gt_2,142,864). coords(wk2gt_2,99,63,864).  visible(wk2gt_2,864).
orientation(wk2gt_1,127,865). coords(wk2gt_1,115,53,865).  visible(wk2gt_1,865). orientation(wk2gt_2,142,865). coords(wk2gt_2,99,63,865).  visible(wk2gt_2,865).
orientation(wk2gt_1,127,866). coords(wk2gt_1,115,53,866).  visible(wk2gt_1,866). orientation(wk2gt_2,142,866). coords(wk2gt_2,99,63,866).  visible(wk2gt_2,866).
orientation(wk2gt_1,127,867). coords(wk2gt_1,115,53,867).  visible(wk2gt_1,867). orientation(wk2gt_2,142,867). coords(wk2gt_2,99,63,867).  visible(wk2gt_2,867).
orientation(wk2gt_1,127,868). coords(wk2gt_1,115,53,868).  visible(wk2gt_1,868). orientation(wk2gt_2,142,868). coords(wk2gt_2,99,63,868).  visible(wk2gt_2,868).
orientation(wk2gt_1,127,869). coords(wk2gt_1,115,53,869).  visible(wk2gt_1,869). orientation(wk2gt_2,142,869). coords(wk2gt_2,99,63,869).  visible(wk2gt_2,869).
orientation(wk2gt_1,127,870). coords(wk2gt_1,115,53,870).  visible(wk2gt_1,870). orientation(wk2gt_2,142,870). coords(wk2gt_2,99,63,870).  visible(wk2gt_2,870).
orientation(wk2gt_1,127,871). coords(wk2gt_1,115,53,871).  visible(wk2gt_1,871). orientation(wk2gt_2,142,871). coords(wk2gt_2,99,63,871).  visible(wk2gt_2,871).
orientation(wk2gt_1,127,872). coords(wk2gt_1,115,53,872).  visible(wk2gt_1,872). orientation(wk2gt_2,142,872). coords(wk2gt_2,99,63,872).  visible(wk2gt_2,872).
orientation(wk2gt_1,127,873). coords(wk2gt_1,115,53,873).  visible(wk2gt_1,873). orientation(wk2gt_2,142,873). coords(wk2gt_2,99,63,873).  visible(wk2gt_2,873).
orientation(wk2gt_1,127,874). coords(wk2gt_1,115,53,874).  visible(wk2gt_1,874). orientation(wk2gt_2,142,874). coords(wk2gt_2,99,63,874).  visible(wk2gt_2,874).
orientation(wk2gt_1,127,875). coords(wk2gt_1,115,53,875).  visible(wk2gt_1,875). orientation(wk2gt_2,142,875). coords(wk2gt_2,99,63,875).  visible(wk2gt_2,875).
orientation(wk2gt_1,127,876). coords(wk2gt_1,115,53,876).  visible(wk2gt_1,876). orientation(wk2gt_2,142,876). coords(wk2gt_2,99,63,876).  visible(wk2gt_2,876).
orientation(wk2gt_1,127,877). coords(wk2gt_1,115,53,877).  visible(wk2gt_1,877). orientation(wk2gt_2,142,877). coords(wk2gt_2,99,63,877).  visible(wk2gt_2,877).
happensAt(disappear(wk2gt_1),876). happensAt(disappear(wk2gt_2),876).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=852..876.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=852..876.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),852). :- holdsAt(meeting(wk2gt_2,wk2gt_1),852).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),853). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),853).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),854). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),854).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),855). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),855).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),856). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),856).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),857). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),857).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),858). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),858).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),859). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),859).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),860). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),860).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),861). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),861).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),862). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),862).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),863). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),863).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),864). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),864).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),865). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),865).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),866). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),866).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),867). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),867).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),868). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),868).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),869). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),869).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),870). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),870).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),871). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),871).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),872). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),872).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),873). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),873).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),874). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),874).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),875). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),875).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),876). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),876).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),877). :- holdsAt(meeting(wk2gt_2,wk2gt_1),877).
:- simpleEvent(T,P,walking), T>852 , T<876.
:- simpleEvent(T,P,running), T>852 , T<876.

#show simpleEvent/3.