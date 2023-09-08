
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

	orientation(fcgt_3,165,264). coords(fcgt_3,100,180,264).  visible(fcgt_3,264). happensAt(walking(fcgt_3),264). orientation(fcgt_4,166,264). coords(fcgt_4,81,205,264).  visible(fcgt_4,264). happensAt(inactive(fcgt_4),264).
orientation(fcgt_3,164,265). coords(fcgt_3,99,180,265).  visible(fcgt_3,265). happensAt(walking(fcgt_3),265). orientation(fcgt_4,166,265). coords(fcgt_4,81,205,265).  visible(fcgt_4,265). happensAt(inactive(fcgt_4),265).
orientation(fcgt_3,163,266). coords(fcgt_3,99,181,266).  visible(fcgt_3,266). happensAt(walking(fcgt_3),266). orientation(fcgt_4,166,266). coords(fcgt_4,81,205,266).  visible(fcgt_4,266). happensAt(inactive(fcgt_4),266).
orientation(fcgt_3,161,267). coords(fcgt_3,99,181,267).  visible(fcgt_3,267). happensAt(walking(fcgt_3),267). orientation(fcgt_4,166,267). coords(fcgt_4,81,205,267).  visible(fcgt_4,267). happensAt(inactive(fcgt_4),267).
orientation(fcgt_3,163,268). coords(fcgt_3,99,181,268).  visible(fcgt_3,268). happensAt(walking(fcgt_3),268). orientation(fcgt_4,166,268). coords(fcgt_4,81,205,268).  visible(fcgt_4,268). happensAt(inactive(fcgt_4),268).
orientation(fcgt_3,166,269). coords(fcgt_3,99,181,269).  visible(fcgt_3,269). happensAt(walking(fcgt_3),269). orientation(fcgt_4,166,269). coords(fcgt_4,81,205,269).  visible(fcgt_4,269). happensAt(inactive(fcgt_4),269).
orientation(fcgt_3,164,270). coords(fcgt_3,98,181,270).  visible(fcgt_3,270). happensAt(walking(fcgt_3),270). orientation(fcgt_4,166,270). coords(fcgt_4,81,205,270).  visible(fcgt_4,270). happensAt(inactive(fcgt_4),270).
orientation(fcgt_3,164,271). coords(fcgt_3,98,181,271).  visible(fcgt_3,271). happensAt(walking(fcgt_3),271). orientation(fcgt_4,166,271). coords(fcgt_4,81,205,271).  visible(fcgt_4,271). happensAt(inactive(fcgt_4),271).
orientation(fcgt_3,164,272). coords(fcgt_3,98,181,272).  visible(fcgt_3,272). happensAt(walking(fcgt_3),272). orientation(fcgt_4,166,272). coords(fcgt_4,81,205,272).  visible(fcgt_4,272). happensAt(inactive(fcgt_4),272).
orientation(fcgt_3,164,273). coords(fcgt_3,98,181,273).  visible(fcgt_3,273). happensAt(walking(fcgt_3),273). orientation(fcgt_4,166,273). coords(fcgt_4,81,205,273).  visible(fcgt_4,273). happensAt(inactive(fcgt_4),273).
orientation(fcgt_3,164,274). coords(fcgt_3,98,181,274).  visible(fcgt_3,274). happensAt(walking(fcgt_3),274). orientation(fcgt_4,166,274). coords(fcgt_4,81,205,274).  visible(fcgt_4,274). happensAt(inactive(fcgt_4),274).
orientation(fcgt_3,164,275). coords(fcgt_3,98,181,275).  visible(fcgt_3,275). happensAt(walking(fcgt_3),275). orientation(fcgt_4,166,275). coords(fcgt_4,81,205,275).  visible(fcgt_4,275). happensAt(inactive(fcgt_4),275).
orientation(fcgt_3,164,276). coords(fcgt_3,98,182,276).  visible(fcgt_3,276). happensAt(walking(fcgt_3),276). orientation(fcgt_4,166,276). coords(fcgt_4,81,205,276).  visible(fcgt_4,276). happensAt(inactive(fcgt_4),276).
orientation(fcgt_3,164,277). coords(fcgt_3,98,182,277).  visible(fcgt_3,277). happensAt(walking(fcgt_3),277). orientation(fcgt_4,166,277). coords(fcgt_4,81,205,277).  visible(fcgt_4,277). happensAt(inactive(fcgt_4),277).
orientation(fcgt_3,164,278). coords(fcgt_3,98,182,278).  visible(fcgt_3,278). happensAt(walking(fcgt_3),278). orientation(fcgt_4,166,278). coords(fcgt_4,81,205,278).  visible(fcgt_4,278). happensAt(inactive(fcgt_4),278).
orientation(fcgt_3,164,279). coords(fcgt_3,98,183,279).  visible(fcgt_3,279). happensAt(walking(fcgt_3),279). orientation(fcgt_4,166,279). coords(fcgt_4,81,205,279).  visible(fcgt_4,279). happensAt(inactive(fcgt_4),279).
orientation(fcgt_3,164,280). coords(fcgt_3,98,183,280).  visible(fcgt_3,280). happensAt(walking(fcgt_3),280). orientation(fcgt_4,166,280). coords(fcgt_4,81,205,280).  visible(fcgt_4,280). happensAt(inactive(fcgt_4),280).
orientation(fcgt_3,164,281). coords(fcgt_3,98,184,281).  visible(fcgt_3,281). happensAt(walking(fcgt_3),281). orientation(fcgt_4,166,281). coords(fcgt_4,81,205,281).  visible(fcgt_4,281). happensAt(inactive(fcgt_4),281).
orientation(fcgt_3,164,282). coords(fcgt_3,98,185,282).  visible(fcgt_3,282). happensAt(walking(fcgt_3),282). orientation(fcgt_4,166,282). coords(fcgt_4,81,205,282).  visible(fcgt_4,282). happensAt(inactive(fcgt_4),282).
orientation(fcgt_3,164,283). coords(fcgt_3,98,185,283).  visible(fcgt_3,283). happensAt(walking(fcgt_3),283). orientation(fcgt_4,166,283). coords(fcgt_4,81,205,283).  visible(fcgt_4,283). happensAt(inactive(fcgt_4),283).
orientation(fcgt_3,164,284). coords(fcgt_3,98,185,284).  visible(fcgt_3,284). happensAt(walking(fcgt_3),284). orientation(fcgt_4,166,284). coords(fcgt_4,82,206,284).  visible(fcgt_4,284). happensAt(inactive(fcgt_4),284).
orientation(fcgt_3,164,285). coords(fcgt_3,98,185,285).  visible(fcgt_3,285). happensAt(walking(fcgt_3),285). orientation(fcgt_4,166,285). coords(fcgt_4,82,206,285).  visible(fcgt_4,285). happensAt(inactive(fcgt_4),285).
orientation(fcgt_3,164,286). coords(fcgt_3,98,185,286).  visible(fcgt_3,286). happensAt(walking(fcgt_3),286). orientation(fcgt_4,166,286). coords(fcgt_4,82,206,286).  visible(fcgt_4,286). happensAt(inactive(fcgt_4),286).
orientation(fcgt_3,164,287). coords(fcgt_3,98,185,287).  visible(fcgt_3,287). happensAt(walking(fcgt_3),287). orientation(fcgt_4,166,287). coords(fcgt_4,82,206,287).  visible(fcgt_4,287). happensAt(inactive(fcgt_4),287).
orientation(fcgt_3,164,288). coords(fcgt_3,98,185,288).  visible(fcgt_3,288). happensAt(inactive(fcgt_3),288). orientation(fcgt_4,166,288). coords(fcgt_4,82,206,288).  visible(fcgt_4,288). happensAt(inactive(fcgt_4),288).
#show holdsAt/2.