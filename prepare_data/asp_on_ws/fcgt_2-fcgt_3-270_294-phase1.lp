
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

	orientation(fcgt_2,157,270). coords(fcgt_2,71,144,270).  visible(fcgt_2,270). happensAt(walking(fcgt_2),270). orientation(fcgt_3,164,270). coords(fcgt_3,98,181,270).  visible(fcgt_3,270). happensAt(walking(fcgt_3),270).
orientation(fcgt_2,157,271). coords(fcgt_2,73,144,271).  visible(fcgt_2,271). happensAt(walking(fcgt_2),271). orientation(fcgt_3,164,271). coords(fcgt_3,98,181,271).  visible(fcgt_3,271). happensAt(walking(fcgt_3),271).
orientation(fcgt_2,157,272). coords(fcgt_2,75,144,272).  visible(fcgt_2,272). happensAt(walking(fcgt_2),272). orientation(fcgt_3,164,272). coords(fcgt_3,98,181,272).  visible(fcgt_3,272). happensAt(walking(fcgt_3),272).
orientation(fcgt_2,157,273). coords(fcgt_2,75,143,273).  visible(fcgt_2,273). happensAt(walking(fcgt_2),273). orientation(fcgt_3,164,273). coords(fcgt_3,98,181,273).  visible(fcgt_3,273). happensAt(walking(fcgt_3),273).
orientation(fcgt_2,157,274). coords(fcgt_2,77,143,274).  visible(fcgt_2,274). happensAt(walking(fcgt_2),274). orientation(fcgt_3,164,274). coords(fcgt_3,98,181,274).  visible(fcgt_3,274). happensAt(walking(fcgt_3),274).
orientation(fcgt_2,157,275). coords(fcgt_2,77,143,275).  visible(fcgt_2,275). happensAt(walking(fcgt_2),275). orientation(fcgt_3,164,275). coords(fcgt_3,98,181,275).  visible(fcgt_3,275). happensAt(walking(fcgt_3),275).
orientation(fcgt_2,157,276). coords(fcgt_2,77,142,276).  visible(fcgt_2,276). happensAt(walking(fcgt_2),276). orientation(fcgt_3,164,276). coords(fcgt_3,98,182,276).  visible(fcgt_3,276). happensAt(walking(fcgt_3),276).
orientation(fcgt_2,157,277). coords(fcgt_2,77,141,277).  visible(fcgt_2,277). happensAt(walking(fcgt_2),277). orientation(fcgt_3,164,277). coords(fcgt_3,98,182,277).  visible(fcgt_3,277). happensAt(walking(fcgt_3),277).
orientation(fcgt_2,157,278). coords(fcgt_2,77,141,278).  visible(fcgt_2,278). happensAt(walking(fcgt_2),278). orientation(fcgt_3,164,278). coords(fcgt_3,98,182,278).  visible(fcgt_3,278). happensAt(walking(fcgt_3),278).
orientation(fcgt_2,157,279). coords(fcgt_2,78,140,279).  visible(fcgt_2,279). happensAt(walking(fcgt_2),279). orientation(fcgt_3,164,279). coords(fcgt_3,98,183,279).  visible(fcgt_3,279). happensAt(walking(fcgt_3),279).
orientation(fcgt_2,150,280). coords(fcgt_2,78,138,280).  visible(fcgt_2,280). happensAt(walking(fcgt_2),280). orientation(fcgt_3,164,280). coords(fcgt_3,98,183,280).  visible(fcgt_3,280). happensAt(walking(fcgt_3),280).
orientation(fcgt_2,150,281). coords(fcgt_2,78,138,281).  visible(fcgt_2,281). happensAt(walking(fcgt_2),281). orientation(fcgt_3,164,281). coords(fcgt_3,98,184,281).  visible(fcgt_3,281). happensAt(walking(fcgt_3),281).
orientation(fcgt_2,150,282). coords(fcgt_2,78,137,282).  visible(fcgt_2,282). happensAt(walking(fcgt_2),282). orientation(fcgt_3,164,282). coords(fcgt_3,98,185,282).  visible(fcgt_3,282). happensAt(walking(fcgt_3),282).
orientation(fcgt_2,150,283). coords(fcgt_2,81,136,283).  visible(fcgt_2,283). happensAt(walking(fcgt_2),283). orientation(fcgt_3,164,283). coords(fcgt_3,98,185,283).  visible(fcgt_3,283). happensAt(walking(fcgt_3),283).
orientation(fcgt_2,150,284). coords(fcgt_2,82,136,284).  visible(fcgt_2,284). happensAt(walking(fcgt_2),284). orientation(fcgt_3,164,284). coords(fcgt_3,98,185,284).  visible(fcgt_3,284). happensAt(walking(fcgt_3),284).
orientation(fcgt_2,150,285). coords(fcgt_2,82,136,285).  visible(fcgt_2,285). happensAt(walking(fcgt_2),285). orientation(fcgt_3,164,285). coords(fcgt_3,98,185,285).  visible(fcgt_3,285). happensAt(walking(fcgt_3),285).
orientation(fcgt_2,150,286). coords(fcgt_2,82,135,286).  visible(fcgt_2,286). happensAt(walking(fcgt_2),286). orientation(fcgt_3,164,286). coords(fcgt_3,98,185,286).  visible(fcgt_3,286). happensAt(walking(fcgt_3),286).
orientation(fcgt_2,150,287). coords(fcgt_2,83,135,287).  visible(fcgt_2,287). happensAt(walking(fcgt_2),287). orientation(fcgt_3,164,287). coords(fcgt_3,98,185,287).  visible(fcgt_3,287). happensAt(walking(fcgt_3),287).
orientation(fcgt_2,150,288). coords(fcgt_2,83,134,288).  visible(fcgt_2,288). happensAt(walking(fcgt_2),288). orientation(fcgt_3,164,288). coords(fcgt_3,98,185,288).  visible(fcgt_3,288). happensAt(inactive(fcgt_3),288).
orientation(fcgt_2,150,289). coords(fcgt_2,83,134,289).  visible(fcgt_2,289). happensAt(walking(fcgt_2),289). orientation(fcgt_3,164,289). coords(fcgt_3,98,186,289).  visible(fcgt_3,289). happensAt(inactive(fcgt_3),289).
orientation(fcgt_2,150,290). coords(fcgt_2,84,133,290).  visible(fcgt_2,290). happensAt(walking(fcgt_2),290). orientation(fcgt_3,164,290). coords(fcgt_3,98,186,290).  visible(fcgt_3,290). happensAt(inactive(fcgt_3),290).
orientation(fcgt_2,150,291). coords(fcgt_2,84,133,291).  visible(fcgt_2,291). happensAt(walking(fcgt_2),291). orientation(fcgt_3,164,291). coords(fcgt_3,98,186,291).  visible(fcgt_3,291). happensAt(inactive(fcgt_3),291).
orientation(fcgt_2,150,292). coords(fcgt_2,85,133,292).  visible(fcgt_2,292). happensAt(walking(fcgt_2),292). orientation(fcgt_3,164,292). coords(fcgt_3,98,186,292).  visible(fcgt_3,292). happensAt(inactive(fcgt_3),292).
orientation(fcgt_2,150,293). coords(fcgt_2,85,133,293).  visible(fcgt_2,293). happensAt(walking(fcgt_2),293). orientation(fcgt_3,164,293). coords(fcgt_3,98,186,293).  visible(fcgt_3,293). happensAt(inactive(fcgt_3),293).
orientation(fcgt_2,150,294). coords(fcgt_2,86,132,294).  visible(fcgt_2,294). happensAt(walking(fcgt_2),294). orientation(fcgt_3,164,294). coords(fcgt_3,98,186,294).  visible(fcgt_3,294). happensAt(inactive(fcgt_3),294).
#show holdsAt/2.