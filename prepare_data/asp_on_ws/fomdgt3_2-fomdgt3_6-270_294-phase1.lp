
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

	orientation(fomdgt3_2,130,270). coords(fomdgt3_2,235,135,270).  visible(fomdgt3_2,270). happensAt(walking(fomdgt3_2),270). orientation(fomdgt3_6,130,270). coords(fomdgt3_6,223,164,270).  visible(fomdgt3_6,270). happensAt(walking(fomdgt3_6),270).
orientation(fomdgt3_2,130,271). coords(fomdgt3_2,235,135,271).  visible(fomdgt3_2,271). happensAt(walking(fomdgt3_2),271). orientation(fomdgt3_6,130,271). coords(fomdgt3_6,223,164,271).  visible(fomdgt3_6,271). happensAt(walking(fomdgt3_6),271).
orientation(fomdgt3_2,130,272). coords(fomdgt3_2,235,135,272).  visible(fomdgt3_2,272). happensAt(walking(fomdgt3_2),272). orientation(fomdgt3_6,130,272). coords(fomdgt3_6,223,164,272).  visible(fomdgt3_6,272). happensAt(walking(fomdgt3_6),272).
orientation(fomdgt3_2,130,273). coords(fomdgt3_2,235,133,273).  visible(fomdgt3_2,273). happensAt(walking(fomdgt3_2),273). orientation(fomdgt3_6,130,273). coords(fomdgt3_6,223,162,273).  visible(fomdgt3_6,273). happensAt(walking(fomdgt3_6),273).
orientation(fomdgt3_2,130,274). coords(fomdgt3_2,236,133,274).  visible(fomdgt3_2,274). happensAt(walking(fomdgt3_2),274). orientation(fomdgt3_6,130,274). coords(fomdgt3_6,223,161,274).  visible(fomdgt3_6,274). happensAt(walking(fomdgt3_6),274).
orientation(fomdgt3_2,130,275). coords(fomdgt3_2,236,133,275).  visible(fomdgt3_2,275). happensAt(walking(fomdgt3_2),275). orientation(fomdgt3_6,130,275). coords(fomdgt3_6,223,162,275).  visible(fomdgt3_6,275). happensAt(walking(fomdgt3_6),275).
orientation(fomdgt3_2,130,276). coords(fomdgt3_2,235,132,276).  visible(fomdgt3_2,276). happensAt(walking(fomdgt3_2),276). orientation(fomdgt3_6,130,276). coords(fomdgt3_6,223,163,276).  visible(fomdgt3_6,276). happensAt(walking(fomdgt3_6),276).
orientation(fomdgt3_2,130,277). coords(fomdgt3_2,235,130,277).  visible(fomdgt3_2,277). happensAt(walking(fomdgt3_2),277). orientation(fomdgt3_6,130,277). coords(fomdgt3_6,223,163,277).  visible(fomdgt3_6,277). happensAt(walking(fomdgt3_6),277).
orientation(fomdgt3_2,130,278). coords(fomdgt3_2,234,129,278).  visible(fomdgt3_2,278). happensAt(walking(fomdgt3_2),278). orientation(fomdgt3_6,130,278). coords(fomdgt3_6,223,159,278).  visible(fomdgt3_6,278). happensAt(walking(fomdgt3_6),278).
orientation(fomdgt3_2,130,279). coords(fomdgt3_2,234,129,279).  visible(fomdgt3_2,279). happensAt(walking(fomdgt3_2),279). orientation(fomdgt3_6,130,279). coords(fomdgt3_6,219,155,279).  visible(fomdgt3_6,279). happensAt(walking(fomdgt3_6),279).
orientation(fomdgt3_2,130,280). coords(fomdgt3_2,234,128,280).  visible(fomdgt3_2,280). happensAt(walking(fomdgt3_2),280). orientation(fomdgt3_6,130,280). coords(fomdgt3_6,219,155,280).  visible(fomdgt3_6,280). happensAt(walking(fomdgt3_6),280).
orientation(fomdgt3_2,130,281). coords(fomdgt3_2,232,127,281).  visible(fomdgt3_2,281). happensAt(walking(fomdgt3_2),281). orientation(fomdgt3_6,130,281). coords(fomdgt3_6,219,153,281).  visible(fomdgt3_6,281). happensAt(walking(fomdgt3_6),281).
orientation(fomdgt3_2,130,282). coords(fomdgt3_2,227,127,282).  visible(fomdgt3_2,282). happensAt(walking(fomdgt3_2),282). orientation(fomdgt3_6,130,282). coords(fomdgt3_6,219,152,282).  visible(fomdgt3_6,282). happensAt(walking(fomdgt3_6),282).
orientation(fomdgt3_2,130,283). coords(fomdgt3_2,227,127,283).  visible(fomdgt3_2,283). happensAt(walking(fomdgt3_2),283). orientation(fomdgt3_6,130,283). coords(fomdgt3_6,223,164,283).  visible(fomdgt3_6,283). happensAt(walking(fomdgt3_6),283).
orientation(fomdgt3_2,130,284). coords(fomdgt3_2,224,127,284).  visible(fomdgt3_2,284). happensAt(walking(fomdgt3_2),284). orientation(fomdgt3_6,130,284). coords(fomdgt3_6,216,155,284).  visible(fomdgt3_6,284). happensAt(walking(fomdgt3_6),284).
orientation(fomdgt3_2,130,285). coords(fomdgt3_2,224,127,285).  visible(fomdgt3_2,285). happensAt(walking(fomdgt3_2),285). orientation(fomdgt3_6,130,285). coords(fomdgt3_6,216,151,285).  visible(fomdgt3_6,285). happensAt(walking(fomdgt3_6),285).
orientation(fomdgt3_2,130,286). coords(fomdgt3_2,224,125,286).  visible(fomdgt3_2,286). happensAt(walking(fomdgt3_2),286). orientation(fomdgt3_6,130,286). coords(fomdgt3_6,216,148,286).  visible(fomdgt3_6,286). happensAt(walking(fomdgt3_6),286).
orientation(fomdgt3_2,130,287). coords(fomdgt3_2,224,123,287).  visible(fomdgt3_2,287). happensAt(walking(fomdgt3_2),287). orientation(fomdgt3_6,130,287). coords(fomdgt3_6,218,144,287).  visible(fomdgt3_6,287). happensAt(walking(fomdgt3_6),287).
orientation(fomdgt3_2,130,288). coords(fomdgt3_2,224,122,288).  visible(fomdgt3_2,288). happensAt(walking(fomdgt3_2),288). orientation(fomdgt3_6,130,288). coords(fomdgt3_6,218,143,288).  visible(fomdgt3_6,288). happensAt(walking(fomdgt3_6),288).
orientation(fomdgt3_2,130,289). coords(fomdgt3_2,224,122,289).  visible(fomdgt3_2,289). happensAt(walking(fomdgt3_2),289). orientation(fomdgt3_6,130,289). coords(fomdgt3_6,219,141,289).  visible(fomdgt3_6,289). happensAt(walking(fomdgt3_6),289).
orientation(fomdgt3_2,130,290). coords(fomdgt3_2,224,121,290).  visible(fomdgt3_2,290). happensAt(walking(fomdgt3_2),290). orientation(fomdgt3_6,130,290). coords(fomdgt3_6,219,141,290).  visible(fomdgt3_6,290). happensAt(walking(fomdgt3_6),290).
orientation(fomdgt3_2,130,291). coords(fomdgt3_2,224,121,291).  visible(fomdgt3_2,291). happensAt(walking(fomdgt3_2),291). orientation(fomdgt3_6,130,291). coords(fomdgt3_6,219,141,291).  visible(fomdgt3_6,291). happensAt(walking(fomdgt3_6),291).
orientation(fomdgt3_2,130,292). coords(fomdgt3_2,223,121,292).  visible(fomdgt3_2,292). happensAt(walking(fomdgt3_2),292). orientation(fomdgt3_6,130,292). coords(fomdgt3_6,221,138,292).  visible(fomdgt3_6,292). happensAt(walking(fomdgt3_6),292).
orientation(fomdgt3_2,130,293). coords(fomdgt3_2,223,121,293).  visible(fomdgt3_2,293). happensAt(walking(fomdgt3_2),293). orientation(fomdgt3_6,130,293). coords(fomdgt3_6,219,136,293).  visible(fomdgt3_6,293). happensAt(walking(fomdgt3_6),293).
orientation(fomdgt3_2,130,294). coords(fomdgt3_2,223,121,294).  visible(fomdgt3_2,294). happensAt(walking(fomdgt3_2),294). orientation(fomdgt3_6,130,294). coords(fomdgt3_6,217,133,294).  visible(fomdgt3_6,294). happensAt(walking(fomdgt3_6),294).
#show holdsAt/2.