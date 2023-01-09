

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

	person(mws1gt_1;mws1gt_3).
orientation(mws1gt_1,155,272). coords(mws1gt_1,97,110,272).  visible(mws1gt_1,272). orientation(mws1gt_3,157,272). coords(mws1gt_3,87,132,272).  visible(mws1gt_3,272).
orientation(mws1gt_1,155,273). coords(mws1gt_1,97,110,273).  visible(mws1gt_1,273). orientation(mws1gt_3,157,273). coords(mws1gt_3,87,132,273).  visible(mws1gt_3,273).
orientation(mws1gt_1,155,274). coords(mws1gt_1,95,109,274).  visible(mws1gt_1,274). orientation(mws1gt_3,155,274). coords(mws1gt_3,87,132,274).  visible(mws1gt_3,274).
orientation(mws1gt_1,155,275). coords(mws1gt_1,95,109,275).  visible(mws1gt_1,275). orientation(mws1gt_3,155,275). coords(mws1gt_3,87,132,275).  visible(mws1gt_3,275).
orientation(mws1gt_1,155,276). coords(mws1gt_1,93,109,276).  visible(mws1gt_1,276). orientation(mws1gt_3,155,276). coords(mws1gt_3,87,132,276).  visible(mws1gt_3,276).
orientation(mws1gt_1,155,277). coords(mws1gt_1,93,109,277).  visible(mws1gt_1,277). orientation(mws1gt_3,155,277). coords(mws1gt_3,86,132,277).  visible(mws1gt_3,277).
orientation(mws1gt_1,155,278). coords(mws1gt_1,93,109,278).  visible(mws1gt_1,278). orientation(mws1gt_3,155,278). coords(mws1gt_3,86,131,278).  visible(mws1gt_3,278).
orientation(mws1gt_1,155,279). coords(mws1gt_1,93,109,279).  visible(mws1gt_1,279). orientation(mws1gt_3,155,279). coords(mws1gt_3,85,131,279).  visible(mws1gt_3,279).
orientation(mws1gt_1,155,280). coords(mws1gt_1,93,108,280).  visible(mws1gt_1,280). orientation(mws1gt_3,155,280). coords(mws1gt_3,85,131,280).  visible(mws1gt_3,280).
orientation(mws1gt_1,155,281). coords(mws1gt_1,93,108,281).  visible(mws1gt_1,281). orientation(mws1gt_3,155,281). coords(mws1gt_3,84,131,281).  visible(mws1gt_3,281).
orientation(mws1gt_1,155,282). coords(mws1gt_1,92,108,282).  visible(mws1gt_1,282). orientation(mws1gt_3,155,282). coords(mws1gt_3,83,131,282).  visible(mws1gt_3,282).
orientation(mws1gt_1,155,283). coords(mws1gt_1,92,108,283).  visible(mws1gt_1,283). orientation(mws1gt_3,155,283). coords(mws1gt_3,82,131,283).  visible(mws1gt_3,283).
orientation(mws1gt_1,155,284). coords(mws1gt_1,92,107,284).  visible(mws1gt_1,284). orientation(mws1gt_3,155,284). coords(mws1gt_3,82,131,284).  visible(mws1gt_3,284).
orientation(mws1gt_1,155,285). coords(mws1gt_1,91,107,285).  visible(mws1gt_1,285). orientation(mws1gt_3,155,285). coords(mws1gt_3,81,130,285).  visible(mws1gt_3,285).
orientation(mws1gt_1,155,286). coords(mws1gt_1,91,106,286).  visible(mws1gt_1,286). orientation(mws1gt_3,155,286). coords(mws1gt_3,80,131,286).  visible(mws1gt_3,286).
orientation(mws1gt_1,155,287). coords(mws1gt_1,89,106,287).  visible(mws1gt_1,287). orientation(mws1gt_3,155,287). coords(mws1gt_3,80,131,287).  visible(mws1gt_3,287).
orientation(mws1gt_1,155,288). coords(mws1gt_1,88,105,288).  visible(mws1gt_1,288). orientation(mws1gt_3,155,288). coords(mws1gt_3,80,131,288).  visible(mws1gt_3,288).
orientation(mws1gt_1,155,289). coords(mws1gt_1,88,105,289).  visible(mws1gt_1,289). orientation(mws1gt_3,155,289). coords(mws1gt_3,80,131,289).  visible(mws1gt_3,289).
orientation(mws1gt_1,155,290). coords(mws1gt_1,87,104,290).  visible(mws1gt_1,290). orientation(mws1gt_3,155,290). coords(mws1gt_3,80,131,290).  visible(mws1gt_3,290).
orientation(mws1gt_1,155,291). coords(mws1gt_1,87,104,291).  visible(mws1gt_1,291). orientation(mws1gt_3,155,291). coords(mws1gt_3,79,131,291).  visible(mws1gt_3,291).
orientation(mws1gt_1,155,292). coords(mws1gt_1,87,104,292).  visible(mws1gt_1,292). orientation(mws1gt_3,155,292). coords(mws1gt_3,79,131,292).  visible(mws1gt_3,292).
orientation(mws1gt_1,155,293). coords(mws1gt_1,87,104,293).  visible(mws1gt_1,293). orientation(mws1gt_3,158,293). coords(mws1gt_3,78,131,293).  visible(mws1gt_3,293).
orientation(mws1gt_1,155,294). coords(mws1gt_1,87,103,294).  visible(mws1gt_1,294). orientation(mws1gt_3,158,294). coords(mws1gt_3,77,131,294).  visible(mws1gt_3,294).
orientation(mws1gt_1,155,295). coords(mws1gt_1,87,103,295).  visible(mws1gt_1,295). orientation(mws1gt_3,158,295). coords(mws1gt_3,76,131,295).  visible(mws1gt_3,295).
orientation(mws1gt_1,155,296). coords(mws1gt_1,87,103,296).  visible(mws1gt_1,296). orientation(mws1gt_3,158,296). coords(mws1gt_3,76,131,296).  visible(mws1gt_3,296).
orientation(mws1gt_1,155,297). coords(mws1gt_1,87,103,297).  visible(mws1gt_1,297). orientation(mws1gt_3,158,297). coords(mws1gt_3,76,131,297).  visible(mws1gt_3,297).
happensAt(disappear(mws1gt_1),296). happensAt(disappear(mws1gt_3),296).
1{simpleEvent(T,mws1gt_1,active); simpleEvent(T,mws1gt_1,inactive); simpleEvent(T,mws1gt_1,walking); simpleEvent(T,mws1gt_1,running)}1 :- T=272..296.
1{simpleEvent(T,mws1gt_3,active); simpleEvent(T,mws1gt_3,inactive); simpleEvent(T,mws1gt_3,walking); simpleEvent(T,mws1gt_3,running)}1 :- T=272..296.
:- holdsAt(moving(mws1gt_1,mws1gt_3),272). :- holdsAt(moving(mws1gt_3,mws1gt_1),272).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),273). :- not holdsAt(moving(mws1gt_3,mws1gt_1),273).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),274). :- not holdsAt(moving(mws1gt_3,mws1gt_1),274).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),275). :- not holdsAt(moving(mws1gt_3,mws1gt_1),275).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),276). :- not holdsAt(moving(mws1gt_3,mws1gt_1),276).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),277). :- not holdsAt(moving(mws1gt_3,mws1gt_1),277).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),278). :- not holdsAt(moving(mws1gt_3,mws1gt_1),278).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),279). :- not holdsAt(moving(mws1gt_3,mws1gt_1),279).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),280). :- not holdsAt(moving(mws1gt_3,mws1gt_1),280).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),281). :- not holdsAt(moving(mws1gt_3,mws1gt_1),281).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),282). :- not holdsAt(moving(mws1gt_3,mws1gt_1),282).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),283). :- not holdsAt(moving(mws1gt_3,mws1gt_1),283).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),284). :- not holdsAt(moving(mws1gt_3,mws1gt_1),284).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),285). :- not holdsAt(moving(mws1gt_3,mws1gt_1),285).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),286). :- not holdsAt(moving(mws1gt_3,mws1gt_1),286).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),287). :- not holdsAt(moving(mws1gt_3,mws1gt_1),287).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),288). :- not holdsAt(moving(mws1gt_3,mws1gt_1),288).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),289). :- not holdsAt(moving(mws1gt_3,mws1gt_1),289).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),290). :- not holdsAt(moving(mws1gt_3,mws1gt_1),290).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),291). :- not holdsAt(moving(mws1gt_3,mws1gt_1),291).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),292). :- not holdsAt(moving(mws1gt_3,mws1gt_1),292).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),293). :- not holdsAt(moving(mws1gt_3,mws1gt_1),293).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),294). :- not holdsAt(moving(mws1gt_3,mws1gt_1),294).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),295). :- not holdsAt(moving(mws1gt_3,mws1gt_1),295).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),296). :- not holdsAt(moving(mws1gt_3,mws1gt_1),296).
:- holdsAt(moving(mws1gt_1,mws1gt_3),297). :- holdsAt(moving(mws1gt_3,mws1gt_1),297).
:- simpleEvent(T,P,inactive), T>272 , T<296.
:- simpleEvent(T,P,running), T>272 , T<296.

#show simpleEvent/3.