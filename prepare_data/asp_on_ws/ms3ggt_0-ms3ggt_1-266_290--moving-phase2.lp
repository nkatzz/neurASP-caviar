

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

	person(ms3ggt_0;ms3ggt_1).
orientation(ms3ggt_0,157,266). coords(ms3ggt_0,63,151,266).  visible(ms3ggt_0,266). orientation(ms3ggt_1,158,266). coords(ms3ggt_1,62,165,266).  visible(ms3ggt_1,266).
orientation(ms3ggt_0,157,267). coords(ms3ggt_0,63,151,267).  visible(ms3ggt_0,267). orientation(ms3ggt_1,158,267). coords(ms3ggt_1,62,165,267).  visible(ms3ggt_1,267).
orientation(ms3ggt_0,157,268). coords(ms3ggt_0,63,151,268).  visible(ms3ggt_0,268). orientation(ms3ggt_1,158,268). coords(ms3ggt_1,62,166,268).  visible(ms3ggt_1,268).
orientation(ms3ggt_0,157,269). coords(ms3ggt_0,63,150,269).  visible(ms3ggt_0,269). orientation(ms3ggt_1,158,269). coords(ms3ggt_1,62,166,269).  visible(ms3ggt_1,269).
orientation(ms3ggt_0,157,270). coords(ms3ggt_0,63,150,270).  visible(ms3ggt_0,270). orientation(ms3ggt_1,158,270). coords(ms3ggt_1,62,166,270).  visible(ms3ggt_1,270).
orientation(ms3ggt_0,157,271). coords(ms3ggt_0,63,150,271).  visible(ms3ggt_0,271). orientation(ms3ggt_1,158,271). coords(ms3ggt_1,62,166,271).  visible(ms3ggt_1,271).
orientation(ms3ggt_0,157,272). coords(ms3ggt_0,65,150,272).  visible(ms3ggt_0,272). orientation(ms3ggt_1,158,272). coords(ms3ggt_1,62,166,272).  visible(ms3ggt_1,272).
orientation(ms3ggt_0,157,273). coords(ms3ggt_0,66,150,273).  visible(ms3ggt_0,273). orientation(ms3ggt_1,158,273). coords(ms3ggt_1,63,165,273).  visible(ms3ggt_1,273).
orientation(ms3ggt_0,161,274). coords(ms3ggt_0,67,150,274).  visible(ms3ggt_0,274). orientation(ms3ggt_1,158,274). coords(ms3ggt_1,63,164,274).  visible(ms3ggt_1,274).
orientation(ms3ggt_0,161,275). coords(ms3ggt_0,67,150,275).  visible(ms3ggt_0,275). orientation(ms3ggt_1,158,275). coords(ms3ggt_1,63,164,275).  visible(ms3ggt_1,275).
orientation(ms3ggt_0,161,276). coords(ms3ggt_0,68,150,276).  visible(ms3ggt_0,276). orientation(ms3ggt_1,158,276). coords(ms3ggt_1,63,164,276).  visible(ms3ggt_1,276).
orientation(ms3ggt_0,161,277). coords(ms3ggt_0,68,150,277).  visible(ms3ggt_0,277). orientation(ms3ggt_1,158,277). coords(ms3ggt_1,66,164,277).  visible(ms3ggt_1,277).
orientation(ms3ggt_0,161,278). coords(ms3ggt_0,68,149,278).  visible(ms3ggt_0,278). orientation(ms3ggt_1,158,278). coords(ms3ggt_1,66,164,278).  visible(ms3ggt_1,278).
orientation(ms3ggt_0,161,279). coords(ms3ggt_0,69,149,279).  visible(ms3ggt_0,279). orientation(ms3ggt_1,158,279). coords(ms3ggt_1,68,164,279).  visible(ms3ggt_1,279).
orientation(ms3ggt_0,161,280). coords(ms3ggt_0,69,149,280).  visible(ms3ggt_0,280). orientation(ms3ggt_1,158,280). coords(ms3ggt_1,68,164,280).  visible(ms3ggt_1,280).
orientation(ms3ggt_0,161,281). coords(ms3ggt_0,69,149,281).  visible(ms3ggt_0,281). orientation(ms3ggt_1,158,281). coords(ms3ggt_1,68,164,281).  visible(ms3ggt_1,281).
orientation(ms3ggt_0,153,282). coords(ms3ggt_0,69,148,282).  visible(ms3ggt_0,282). orientation(ms3ggt_1,158,282). coords(ms3ggt_1,69,164,282).  visible(ms3ggt_1,282).
orientation(ms3ggt_0,153,283). coords(ms3ggt_0,69,148,283).  visible(ms3ggt_0,283). orientation(ms3ggt_1,158,283). coords(ms3ggt_1,69,164,283).  visible(ms3ggt_1,283).
orientation(ms3ggt_0,153,284). coords(ms3ggt_0,69,148,284).  visible(ms3ggt_0,284). orientation(ms3ggt_1,158,284). coords(ms3ggt_1,69,164,284).  visible(ms3ggt_1,284).
orientation(ms3ggt_0,153,285). coords(ms3ggt_0,69,148,285).  visible(ms3ggt_0,285). orientation(ms3ggt_1,158,285). coords(ms3ggt_1,70,164,285).  visible(ms3ggt_1,285).
orientation(ms3ggt_0,153,286). coords(ms3ggt_0,69,146,286).  visible(ms3ggt_0,286). orientation(ms3ggt_1,158,286). coords(ms3ggt_1,70,164,286).  visible(ms3ggt_1,286).
orientation(ms3ggt_0,153,287). coords(ms3ggt_0,73,146,287).  visible(ms3ggt_0,287). orientation(ms3ggt_1,158,287). coords(ms3ggt_1,70,164,287).  visible(ms3ggt_1,287).
orientation(ms3ggt_0,153,288). coords(ms3ggt_0,74,146,288).  visible(ms3ggt_0,288). orientation(ms3ggt_1,158,288). coords(ms3ggt_1,70,164,288).  visible(ms3ggt_1,288).
orientation(ms3ggt_0,153,289). coords(ms3ggt_0,74,146,289).  visible(ms3ggt_0,289). orientation(ms3ggt_1,158,289). coords(ms3ggt_1,71,164,289).  visible(ms3ggt_1,289).
orientation(ms3ggt_0,159,290). coords(ms3ggt_0,76,146,290).  visible(ms3ggt_0,290). orientation(ms3ggt_1,158,290). coords(ms3ggt_1,72,164,290).  visible(ms3ggt_1,290).
orientation(ms3ggt_0,159,291). coords(ms3ggt_0,76,146,291).  visible(ms3ggt_0,291). orientation(ms3ggt_1,158,291). coords(ms3ggt_1,72,164,291).  visible(ms3ggt_1,291).
happensAt(disappear(ms3ggt_0),290). happensAt(disappear(ms3ggt_1),290).
1{simpleEvent(T,ms3ggt_0,active); simpleEvent(T,ms3ggt_0,inactive); simpleEvent(T,ms3ggt_0,walking); simpleEvent(T,ms3ggt_0,running)}1 :- T=266..290.
1{simpleEvent(T,ms3ggt_1,active); simpleEvent(T,ms3ggt_1,inactive); simpleEvent(T,ms3ggt_1,walking); simpleEvent(T,ms3ggt_1,running)}1 :- T=266..290.
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),266). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),266).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),267). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),267).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),268). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),268).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),269). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),269).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),270). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),270).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),271). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),271).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),272). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),272).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),273). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),273).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),274). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),274).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),275). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),275).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),276). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),276).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),277). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),277).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),278). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),278).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),279). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),279).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),280). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),280).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),281). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),281).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),282). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),282).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),283). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),283).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),284). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),284).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),285). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),285).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),286). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),286).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),287). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),287).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),288). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),288).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),289). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),289).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),290). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),290).
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),291). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),291).
:- simpleEvent(T,P,inactive), T>266 , T<290.
:- simpleEvent(T,P,running), T>266 , T<290.

#show simpleEvent/3.