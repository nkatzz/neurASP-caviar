
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

	orientation(fomdgt3_1,150,270). coords(fomdgt3_1,57,96,270).  visible(fomdgt3_1,270). happensAt(walking(fomdgt3_1),270). orientation(fomdgt3_6,130,270). coords(fomdgt3_6,223,164,270).  visible(fomdgt3_6,270). happensAt(walking(fomdgt3_6),270).
orientation(fomdgt3_1,150,271). coords(fomdgt3_1,57,96,271).  visible(fomdgt3_1,271). happensAt(walking(fomdgt3_1),271). orientation(fomdgt3_6,130,271). coords(fomdgt3_6,223,164,271).  visible(fomdgt3_6,271). happensAt(walking(fomdgt3_6),271).
orientation(fomdgt3_1,150,272). coords(fomdgt3_1,57,96,272).  visible(fomdgt3_1,272). happensAt(walking(fomdgt3_1),272). orientation(fomdgt3_6,130,272). coords(fomdgt3_6,223,164,272).  visible(fomdgt3_6,272). happensAt(walking(fomdgt3_6),272).
orientation(fomdgt3_1,150,273). coords(fomdgt3_1,53,96,273).  visible(fomdgt3_1,273). happensAt(walking(fomdgt3_1),273). orientation(fomdgt3_6,130,273). coords(fomdgt3_6,223,162,273).  visible(fomdgt3_6,273). happensAt(walking(fomdgt3_6),273).
orientation(fomdgt3_1,150,274). coords(fomdgt3_1,53,96,274).  visible(fomdgt3_1,274). happensAt(walking(fomdgt3_1),274). orientation(fomdgt3_6,130,274). coords(fomdgt3_6,223,161,274).  visible(fomdgt3_6,274). happensAt(walking(fomdgt3_6),274).
orientation(fomdgt3_1,150,275). coords(fomdgt3_1,53,96,275).  visible(fomdgt3_1,275). happensAt(walking(fomdgt3_1),275). orientation(fomdgt3_6,130,275). coords(fomdgt3_6,223,162,275).  visible(fomdgt3_6,275). happensAt(walking(fomdgt3_6),275).
orientation(fomdgt3_1,150,276). coords(fomdgt3_1,53,96,276).  visible(fomdgt3_1,276). happensAt(walking(fomdgt3_1),276). orientation(fomdgt3_6,130,276). coords(fomdgt3_6,223,163,276).  visible(fomdgt3_6,276). happensAt(walking(fomdgt3_6),276).
orientation(fomdgt3_1,150,277). coords(fomdgt3_1,51,96,277).  visible(fomdgt3_1,277). happensAt(walking(fomdgt3_1),277). orientation(fomdgt3_6,130,277). coords(fomdgt3_6,223,163,277).  visible(fomdgt3_6,277). happensAt(walking(fomdgt3_6),277).
orientation(fomdgt3_1,150,278). coords(fomdgt3_1,51,96,278).  visible(fomdgt3_1,278). happensAt(walking(fomdgt3_1),278). orientation(fomdgt3_6,130,278). coords(fomdgt3_6,223,159,278).  visible(fomdgt3_6,278). happensAt(walking(fomdgt3_6),278).
orientation(fomdgt3_1,150,279). coords(fomdgt3_1,51,96,279).  visible(fomdgt3_1,279). happensAt(walking(fomdgt3_1),279). orientation(fomdgt3_6,130,279). coords(fomdgt3_6,219,155,279).  visible(fomdgt3_6,279). happensAt(walking(fomdgt3_6),279).
orientation(fomdgt3_1,150,280). coords(fomdgt3_1,51,96,280).  visible(fomdgt3_1,280). happensAt(walking(fomdgt3_1),280). orientation(fomdgt3_6,130,280). coords(fomdgt3_6,219,155,280).  visible(fomdgt3_6,280). happensAt(walking(fomdgt3_6),280).
orientation(fomdgt3_1,150,281). coords(fomdgt3_1,51,96,281).  visible(fomdgt3_1,281). happensAt(walking(fomdgt3_1),281). orientation(fomdgt3_6,130,281). coords(fomdgt3_6,219,153,281).  visible(fomdgt3_6,281). happensAt(walking(fomdgt3_6),281).
orientation(fomdgt3_1,150,282). coords(fomdgt3_1,51,96,282).  visible(fomdgt3_1,282). happensAt(walking(fomdgt3_1),282). orientation(fomdgt3_6,130,282). coords(fomdgt3_6,219,152,282).  visible(fomdgt3_6,282). happensAt(walking(fomdgt3_6),282).
orientation(fomdgt3_1,150,283). coords(fomdgt3_1,51,96,283).  visible(fomdgt3_1,283). happensAt(walking(fomdgt3_1),283). orientation(fomdgt3_6,130,283). coords(fomdgt3_6,223,164,283).  visible(fomdgt3_6,283). happensAt(walking(fomdgt3_6),283).
orientation(fomdgt3_1,150,284). coords(fomdgt3_1,51,96,284).  visible(fomdgt3_1,284). happensAt(walking(fomdgt3_1),284). orientation(fomdgt3_6,130,284). coords(fomdgt3_6,216,155,284).  visible(fomdgt3_6,284). happensAt(walking(fomdgt3_6),284).
orientation(fomdgt3_1,150,285). coords(fomdgt3_1,51,96,285).  visible(fomdgt3_1,285). happensAt(walking(fomdgt3_1),285). orientation(fomdgt3_6,130,285). coords(fomdgt3_6,216,151,285).  visible(fomdgt3_6,285). happensAt(walking(fomdgt3_6),285).
orientation(fomdgt3_1,150,286). coords(fomdgt3_1,51,96,286).  visible(fomdgt3_1,286). happensAt(active(fomdgt3_1),286). orientation(fomdgt3_6,130,286). coords(fomdgt3_6,216,148,286).  visible(fomdgt3_6,286). happensAt(walking(fomdgt3_6),286).
orientation(fomdgt3_1,150,287). coords(fomdgt3_1,51,96,287).  visible(fomdgt3_1,287). happensAt(active(fomdgt3_1),287). orientation(fomdgt3_6,130,287). coords(fomdgt3_6,218,144,287).  visible(fomdgt3_6,287). happensAt(walking(fomdgt3_6),287).
orientation(fomdgt3_1,150,288). coords(fomdgt3_1,51,96,288).  visible(fomdgt3_1,288). happensAt(active(fomdgt3_1),288). orientation(fomdgt3_6,130,288). coords(fomdgt3_6,218,143,288).  visible(fomdgt3_6,288). happensAt(walking(fomdgt3_6),288).
orientation(fomdgt3_1,150,289). coords(fomdgt3_1,46,96,289).  visible(fomdgt3_1,289). happensAt(active(fomdgt3_1),289). orientation(fomdgt3_6,130,289). coords(fomdgt3_6,219,141,289).  visible(fomdgt3_6,289). happensAt(walking(fomdgt3_6),289).
orientation(fomdgt3_1,150,290). coords(fomdgt3_1,46,96,290).  visible(fomdgt3_1,290). happensAt(active(fomdgt3_1),290). orientation(fomdgt3_6,130,290). coords(fomdgt3_6,219,141,290).  visible(fomdgt3_6,290). happensAt(walking(fomdgt3_6),290).
orientation(fomdgt3_1,150,291). coords(fomdgt3_1,46,96,291).  visible(fomdgt3_1,291). happensAt(active(fomdgt3_1),291). orientation(fomdgt3_6,130,291). coords(fomdgt3_6,219,141,291).  visible(fomdgt3_6,291). happensAt(walking(fomdgt3_6),291).
orientation(fomdgt3_1,150,292). coords(fomdgt3_1,46,96,292).  visible(fomdgt3_1,292). happensAt(active(fomdgt3_1),292). orientation(fomdgt3_6,130,292). coords(fomdgt3_6,221,138,292).  visible(fomdgt3_6,292). happensAt(walking(fomdgt3_6),292).
orientation(fomdgt3_1,150,293). coords(fomdgt3_1,46,96,293).  visible(fomdgt3_1,293). happensAt(active(fomdgt3_1),293). orientation(fomdgt3_6,130,293). coords(fomdgt3_6,219,136,293).  visible(fomdgt3_6,293). happensAt(walking(fomdgt3_6),293).
orientation(fomdgt3_1,150,294). coords(fomdgt3_1,46,96,294).  visible(fomdgt3_1,294). happensAt(active(fomdgt3_1),294). orientation(fomdgt3_6,130,294). coords(fomdgt3_6,217,133,294).  visible(fomdgt3_6,294). happensAt(walking(fomdgt3_6),294).
#show holdsAt/2.