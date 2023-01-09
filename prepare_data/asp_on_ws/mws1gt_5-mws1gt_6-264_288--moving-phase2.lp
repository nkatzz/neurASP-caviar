

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

	person(mws1gt_5;mws1gt_6).
orientation(mws1gt_5,161,264). coords(mws1gt_5,41,151,264).  visible(mws1gt_5,264). orientation(mws1gt_6,146,264). coords(mws1gt_6,42,129,264).  visible(mws1gt_6,264).
orientation(mws1gt_5,161,265). coords(mws1gt_5,41,153,265).  visible(mws1gt_5,265). orientation(mws1gt_6,146,265). coords(mws1gt_6,42,129,265).  visible(mws1gt_6,265).
orientation(mws1gt_5,161,266). coords(mws1gt_5,41,153,266).  visible(mws1gt_5,266). orientation(mws1gt_6,146,266). coords(mws1gt_6,42,130,266).  visible(mws1gt_6,266).
orientation(mws1gt_5,161,267). coords(mws1gt_5,41,153,267).  visible(mws1gt_5,267). orientation(mws1gt_6,146,267). coords(mws1gt_6,42,130,267).  visible(mws1gt_6,267).
orientation(mws1gt_5,161,268). coords(mws1gt_5,41,153,268).  visible(mws1gt_5,268). orientation(mws1gt_6,146,268). coords(mws1gt_6,42,130,268).  visible(mws1gt_6,268).
orientation(mws1gt_5,161,269). coords(mws1gt_5,41,153,269).  visible(mws1gt_5,269). orientation(mws1gt_6,146,269). coords(mws1gt_6,42,131,269).  visible(mws1gt_6,269).
orientation(mws1gt_5,161,270). coords(mws1gt_5,40,153,270).  visible(mws1gt_5,270). orientation(mws1gt_6,146,270). coords(mws1gt_6,42,132,270).  visible(mws1gt_6,270).
orientation(mws1gt_5,161,271). coords(mws1gt_5,40,154,271).  visible(mws1gt_5,271). orientation(mws1gt_6,146,271). coords(mws1gt_6,41,133,271).  visible(mws1gt_6,271).
orientation(mws1gt_5,161,272). coords(mws1gt_5,40,154,272).  visible(mws1gt_5,272). orientation(mws1gt_6,146,272). coords(mws1gt_6,41,133,272).  visible(mws1gt_6,272).
orientation(mws1gt_5,161,273). coords(mws1gt_5,40,155,273).  visible(mws1gt_5,273). orientation(mws1gt_6,146,273). coords(mws1gt_6,40,134,273).  visible(mws1gt_6,273).
orientation(mws1gt_5,161,274). coords(mws1gt_5,40,156,274).  visible(mws1gt_5,274). orientation(mws1gt_6,146,274). coords(mws1gt_6,40,134,274).  visible(mws1gt_6,274).
orientation(mws1gt_5,161,275). coords(mws1gt_5,40,157,275).  visible(mws1gt_5,275). orientation(mws1gt_6,146,275). coords(mws1gt_6,40,135,275).  visible(mws1gt_6,275).
orientation(mws1gt_5,161,276). coords(mws1gt_5,39,158,276).  visible(mws1gt_5,276). orientation(mws1gt_6,146,276). coords(mws1gt_6,40,136,276).  visible(mws1gt_6,276).
orientation(mws1gt_5,161,277). coords(mws1gt_5,39,159,277).  visible(mws1gt_5,277). orientation(mws1gt_6,146,277). coords(mws1gt_6,40,137,277).  visible(mws1gt_6,277).
orientation(mws1gt_5,161,278). coords(mws1gt_5,38,160,278).  visible(mws1gt_5,278). orientation(mws1gt_6,146,278). coords(mws1gt_6,40,138,278).  visible(mws1gt_6,278).
orientation(mws1gt_5,161,279). coords(mws1gt_5,38,162,279).  visible(mws1gt_5,279). orientation(mws1gt_6,146,279). coords(mws1gt_6,40,139,279).  visible(mws1gt_6,279).
orientation(mws1gt_5,161,280). coords(mws1gt_5,38,162,280).  visible(mws1gt_5,280). orientation(mws1gt_6,146,280). coords(mws1gt_6,40,140,280).  visible(mws1gt_6,280).
orientation(mws1gt_5,161,281). coords(mws1gt_5,38,163,281).  visible(mws1gt_5,281). orientation(mws1gt_6,146,281). coords(mws1gt_6,40,140,281).  visible(mws1gt_6,281).
orientation(mws1gt_5,161,282). coords(mws1gt_5,38,164,282).  visible(mws1gt_5,282). orientation(mws1gt_6,146,282). coords(mws1gt_6,39,140,282).  visible(mws1gt_6,282).
orientation(mws1gt_5,161,283). coords(mws1gt_5,38,164,283).  visible(mws1gt_5,283). orientation(mws1gt_6,146,283). coords(mws1gt_6,38,142,283).  visible(mws1gt_6,283).
orientation(mws1gt_5,161,284). coords(mws1gt_5,37,164,284).  visible(mws1gt_5,284). orientation(mws1gt_6,146,284). coords(mws1gt_6,38,143,284).  visible(mws1gt_6,284).
orientation(mws1gt_5,161,285). coords(mws1gt_5,37,165,285).  visible(mws1gt_5,285). orientation(mws1gt_6,146,285). coords(mws1gt_6,38,144,285).  visible(mws1gt_6,285).
orientation(mws1gt_5,161,286). coords(mws1gt_5,37,165,286).  visible(mws1gt_5,286). orientation(mws1gt_6,146,286). coords(mws1gt_6,38,145,286).  visible(mws1gt_6,286).
orientation(mws1gt_5,161,287). coords(mws1gt_5,37,166,287).  visible(mws1gt_5,287). orientation(mws1gt_6,146,287). coords(mws1gt_6,38,146,287).  visible(mws1gt_6,287).
orientation(mws1gt_5,161,288). coords(mws1gt_5,37,166,288).  visible(mws1gt_5,288). orientation(mws1gt_6,146,288). coords(mws1gt_6,38,146,288).  visible(mws1gt_6,288).
orientation(mws1gt_5,161,289). coords(mws1gt_5,37,166,289).  visible(mws1gt_5,289). orientation(mws1gt_6,146,289). coords(mws1gt_6,38,146,289).  visible(mws1gt_6,289).
happensAt(disappear(mws1gt_5),288). happensAt(disappear(mws1gt_6),288).
1{simpleEvent(T,mws1gt_5,active); simpleEvent(T,mws1gt_5,inactive); simpleEvent(T,mws1gt_5,walking); simpleEvent(T,mws1gt_5,running)}1 :- T=264..288.
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=264..288.
:- holdsAt(moving(mws1gt_5,mws1gt_6),264). :- holdsAt(moving(mws1gt_6,mws1gt_5),264).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),265). :- not holdsAt(moving(mws1gt_6,mws1gt_5),265).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),266). :- not holdsAt(moving(mws1gt_6,mws1gt_5),266).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),267). :- not holdsAt(moving(mws1gt_6,mws1gt_5),267).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),268). :- not holdsAt(moving(mws1gt_6,mws1gt_5),268).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),269). :- not holdsAt(moving(mws1gt_6,mws1gt_5),269).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),270). :- not holdsAt(moving(mws1gt_6,mws1gt_5),270).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),271). :- not holdsAt(moving(mws1gt_6,mws1gt_5),271).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),272). :- not holdsAt(moving(mws1gt_6,mws1gt_5),272).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),273). :- not holdsAt(moving(mws1gt_6,mws1gt_5),273).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),274). :- not holdsAt(moving(mws1gt_6,mws1gt_5),274).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),275). :- not holdsAt(moving(mws1gt_6,mws1gt_5),275).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),276). :- not holdsAt(moving(mws1gt_6,mws1gt_5),276).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),277). :- not holdsAt(moving(mws1gt_6,mws1gt_5),277).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),278). :- not holdsAt(moving(mws1gt_6,mws1gt_5),278).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),279). :- not holdsAt(moving(mws1gt_6,mws1gt_5),279).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),280). :- not holdsAt(moving(mws1gt_6,mws1gt_5),280).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),281). :- not holdsAt(moving(mws1gt_6,mws1gt_5),281).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),282). :- not holdsAt(moving(mws1gt_6,mws1gt_5),282).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),283). :- not holdsAt(moving(mws1gt_6,mws1gt_5),283).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),284). :- not holdsAt(moving(mws1gt_6,mws1gt_5),284).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),285). :- not holdsAt(moving(mws1gt_6,mws1gt_5),285).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),286). :- not holdsAt(moving(mws1gt_6,mws1gt_5),286).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),287). :- not holdsAt(moving(mws1gt_6,mws1gt_5),287).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),288). :- not holdsAt(moving(mws1gt_6,mws1gt_5),288).
:- holdsAt(moving(mws1gt_5,mws1gt_6),289). :- holdsAt(moving(mws1gt_6,mws1gt_5),289).
:- simpleEvent(T,P,inactive), T>264 , T<288.
:- simpleEvent(T,P,running), T>264 , T<288.

#show simpleEvent/3.