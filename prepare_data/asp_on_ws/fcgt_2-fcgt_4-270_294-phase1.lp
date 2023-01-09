
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

	orientation(fcgt_2,157,270). coords(fcgt_2,71,144,270).  visible(fcgt_2,270). happensAt(walking(fcgt_2),270). orientation(fcgt_4,166,270). coords(fcgt_4,81,205,270).  visible(fcgt_4,270). happensAt(inactive(fcgt_4),270).
orientation(fcgt_2,157,271). coords(fcgt_2,73,144,271).  visible(fcgt_2,271). happensAt(walking(fcgt_2),271). orientation(fcgt_4,166,271). coords(fcgt_4,81,205,271).  visible(fcgt_4,271). happensAt(inactive(fcgt_4),271).
orientation(fcgt_2,157,272). coords(fcgt_2,75,144,272).  visible(fcgt_2,272). happensAt(walking(fcgt_2),272). orientation(fcgt_4,166,272). coords(fcgt_4,81,205,272).  visible(fcgt_4,272). happensAt(inactive(fcgt_4),272).
orientation(fcgt_2,157,273). coords(fcgt_2,75,143,273).  visible(fcgt_2,273). happensAt(walking(fcgt_2),273). orientation(fcgt_4,166,273). coords(fcgt_4,81,205,273).  visible(fcgt_4,273). happensAt(inactive(fcgt_4),273).
orientation(fcgt_2,157,274). coords(fcgt_2,77,143,274).  visible(fcgt_2,274). happensAt(walking(fcgt_2),274). orientation(fcgt_4,166,274). coords(fcgt_4,81,205,274).  visible(fcgt_4,274). happensAt(inactive(fcgt_4),274).
orientation(fcgt_2,157,275). coords(fcgt_2,77,143,275).  visible(fcgt_2,275). happensAt(walking(fcgt_2),275). orientation(fcgt_4,166,275). coords(fcgt_4,81,205,275).  visible(fcgt_4,275). happensAt(inactive(fcgt_4),275).
orientation(fcgt_2,157,276). coords(fcgt_2,77,142,276).  visible(fcgt_2,276). happensAt(walking(fcgt_2),276). orientation(fcgt_4,166,276). coords(fcgt_4,81,205,276).  visible(fcgt_4,276). happensAt(inactive(fcgt_4),276).
orientation(fcgt_2,157,277). coords(fcgt_2,77,141,277).  visible(fcgt_2,277). happensAt(walking(fcgt_2),277). orientation(fcgt_4,166,277). coords(fcgt_4,81,205,277).  visible(fcgt_4,277). happensAt(inactive(fcgt_4),277).
orientation(fcgt_2,157,278). coords(fcgt_2,77,141,278).  visible(fcgt_2,278). happensAt(walking(fcgt_2),278). orientation(fcgt_4,166,278). coords(fcgt_4,81,205,278).  visible(fcgt_4,278). happensAt(inactive(fcgt_4),278).
orientation(fcgt_2,157,279). coords(fcgt_2,78,140,279).  visible(fcgt_2,279). happensAt(walking(fcgt_2),279). orientation(fcgt_4,166,279). coords(fcgt_4,81,205,279).  visible(fcgt_4,279). happensAt(inactive(fcgt_4),279).
orientation(fcgt_2,150,280). coords(fcgt_2,78,138,280).  visible(fcgt_2,280). happensAt(walking(fcgt_2),280). orientation(fcgt_4,166,280). coords(fcgt_4,81,205,280).  visible(fcgt_4,280). happensAt(inactive(fcgt_4),280).
orientation(fcgt_2,150,281). coords(fcgt_2,78,138,281).  visible(fcgt_2,281). happensAt(walking(fcgt_2),281). orientation(fcgt_4,166,281). coords(fcgt_4,81,205,281).  visible(fcgt_4,281). happensAt(inactive(fcgt_4),281).
orientation(fcgt_2,150,282). coords(fcgt_2,78,137,282).  visible(fcgt_2,282). happensAt(walking(fcgt_2),282). orientation(fcgt_4,166,282). coords(fcgt_4,81,205,282).  visible(fcgt_4,282). happensAt(inactive(fcgt_4),282).
orientation(fcgt_2,150,283). coords(fcgt_2,81,136,283).  visible(fcgt_2,283). happensAt(walking(fcgt_2),283). orientation(fcgt_4,166,283). coords(fcgt_4,81,205,283).  visible(fcgt_4,283). happensAt(inactive(fcgt_4),283).
orientation(fcgt_2,150,284). coords(fcgt_2,82,136,284).  visible(fcgt_2,284). happensAt(walking(fcgt_2),284). orientation(fcgt_4,166,284). coords(fcgt_4,82,206,284).  visible(fcgt_4,284). happensAt(inactive(fcgt_4),284).
orientation(fcgt_2,150,285). coords(fcgt_2,82,136,285).  visible(fcgt_2,285). happensAt(walking(fcgt_2),285). orientation(fcgt_4,166,285). coords(fcgt_4,82,206,285).  visible(fcgt_4,285). happensAt(inactive(fcgt_4),285).
orientation(fcgt_2,150,286). coords(fcgt_2,82,135,286).  visible(fcgt_2,286). happensAt(walking(fcgt_2),286). orientation(fcgt_4,166,286). coords(fcgt_4,82,206,286).  visible(fcgt_4,286). happensAt(inactive(fcgt_4),286).
orientation(fcgt_2,150,287). coords(fcgt_2,83,135,287).  visible(fcgt_2,287). happensAt(walking(fcgt_2),287). orientation(fcgt_4,166,287). coords(fcgt_4,82,206,287).  visible(fcgt_4,287). happensAt(inactive(fcgt_4),287).
orientation(fcgt_2,150,288). coords(fcgt_2,83,134,288).  visible(fcgt_2,288). happensAt(walking(fcgt_2),288). orientation(fcgt_4,166,288). coords(fcgt_4,82,206,288).  visible(fcgt_4,288). happensAt(inactive(fcgt_4),288).
orientation(fcgt_2,150,289). coords(fcgt_2,83,134,289).  visible(fcgt_2,289). happensAt(walking(fcgt_2),289). orientation(fcgt_4,166,289). coords(fcgt_4,82,206,289).  visible(fcgt_4,289). happensAt(inactive(fcgt_4),289).
orientation(fcgt_2,150,290). coords(fcgt_2,84,133,290).  visible(fcgt_2,290). happensAt(walking(fcgt_2),290). orientation(fcgt_4,166,290). coords(fcgt_4,82,206,290).  visible(fcgt_4,290). happensAt(inactive(fcgt_4),290).
orientation(fcgt_2,150,291). coords(fcgt_2,84,133,291).  visible(fcgt_2,291). happensAt(walking(fcgt_2),291). orientation(fcgt_4,166,291). coords(fcgt_4,82,206,291).  visible(fcgt_4,291). happensAt(inactive(fcgt_4),291).
orientation(fcgt_2,150,292). coords(fcgt_2,85,133,292).  visible(fcgt_2,292). happensAt(walking(fcgt_2),292). orientation(fcgt_4,166,292). coords(fcgt_4,82,206,292).  visible(fcgt_4,292). happensAt(inactive(fcgt_4),292).
orientation(fcgt_2,150,293). coords(fcgt_2,85,133,293).  visible(fcgt_2,293). happensAt(walking(fcgt_2),293). orientation(fcgt_4,166,293). coords(fcgt_4,82,206,293).  visible(fcgt_4,293). happensAt(inactive(fcgt_4),293).
orientation(fcgt_2,150,294). coords(fcgt_2,86,132,294).  visible(fcgt_2,294). happensAt(walking(fcgt_2),294). orientation(fcgt_4,166,294). coords(fcgt_4,82,206,294).  visible(fcgt_4,294). happensAt(inactive(fcgt_4),294).
#show holdsAt/2.