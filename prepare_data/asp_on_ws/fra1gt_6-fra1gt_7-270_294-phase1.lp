
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

	orientation(fra1gt_6,123,270). coords(fra1gt_6,218,60,270).  visible(fra1gt_6,270). happensAt(walking(fra1gt_6),270). orientation(fra1gt_7,137,270). coords(fra1gt_7,142,45,270).  visible(fra1gt_7,270). happensAt(walking(fra1gt_7),270).
orientation(fra1gt_6,123,271). coords(fra1gt_6,217,61,271).  visible(fra1gt_6,271). happensAt(walking(fra1gt_6),271). orientation(fra1gt_7,137,271). coords(fra1gt_7,142,45,271).  visible(fra1gt_7,271). happensAt(walking(fra1gt_7),271).
orientation(fra1gt_6,123,272). coords(fra1gt_6,216,61,272).  visible(fra1gt_6,272). happensAt(walking(fra1gt_6),272). orientation(fra1gt_7,137,272). coords(fra1gt_7,142,45,272).  visible(fra1gt_7,272). happensAt(walking(fra1gt_7),272).
orientation(fra1gt_6,125,273). coords(fra1gt_6,215,61,273).  visible(fra1gt_6,273). happensAt(walking(fra1gt_6),273). orientation(fra1gt_7,137,273). coords(fra1gt_7,142,45,273).  visible(fra1gt_7,273). happensAt(walking(fra1gt_7),273).
orientation(fra1gt_6,128,274). coords(fra1gt_6,213,62,274).  visible(fra1gt_6,274). happensAt(walking(fra1gt_6),274). orientation(fra1gt_7,137,274). coords(fra1gt_7,142,45,274).  visible(fra1gt_7,274). happensAt(walking(fra1gt_7),274).
orientation(fra1gt_6,127,275). coords(fra1gt_6,211,62,275).  visible(fra1gt_6,275). happensAt(walking(fra1gt_6),275). orientation(fra1gt_7,137,275). coords(fra1gt_7,141,45,275).  visible(fra1gt_7,275). happensAt(walking(fra1gt_7),275).
orientation(fra1gt_6,128,276). coords(fra1gt_6,210,63,276).  visible(fra1gt_6,276). happensAt(walking(fra1gt_6),276). orientation(fra1gt_7,137,276). coords(fra1gt_7,141,46,276).  visible(fra1gt_7,276). happensAt(walking(fra1gt_7),276).
orientation(fra1gt_6,128,277). coords(fra1gt_6,208,63,277).  visible(fra1gt_6,277). happensAt(walking(fra1gt_6),277). orientation(fra1gt_7,137,277). coords(fra1gt_7,141,46,277).  visible(fra1gt_7,277). happensAt(walking(fra1gt_7),277).
orientation(fra1gt_6,128,278). coords(fra1gt_6,206,64,278).  visible(fra1gt_6,278). happensAt(walking(fra1gt_6),278). orientation(fra1gt_7,137,278). coords(fra1gt_7,141,46,278).  visible(fra1gt_7,278). happensAt(walking(fra1gt_7),278).
orientation(fra1gt_6,128,279). coords(fra1gt_6,204,66,279).  visible(fra1gt_6,279). happensAt(walking(fra1gt_6),279). orientation(fra1gt_7,137,279). coords(fra1gt_7,141,46,279).  visible(fra1gt_7,279). happensAt(walking(fra1gt_7),279).
orientation(fra1gt_6,128,280). coords(fra1gt_6,203,68,280).  visible(fra1gt_6,280). happensAt(walking(fra1gt_6),280). orientation(fra1gt_7,137,280). coords(fra1gt_7,140,46,280).  visible(fra1gt_7,280). happensAt(walking(fra1gt_7),280).
orientation(fra1gt_6,128,281). coords(fra1gt_6,203,69,281).  visible(fra1gt_6,281). happensAt(walking(fra1gt_6),281). orientation(fra1gt_7,137,281). coords(fra1gt_7,140,47,281).  visible(fra1gt_7,281). happensAt(walking(fra1gt_7),281).
orientation(fra1gt_6,128,282). coords(fra1gt_6,202,70,282).  visible(fra1gt_6,282). happensAt(walking(fra1gt_6),282). orientation(fra1gt_7,137,282). coords(fra1gt_7,140,47,282).  visible(fra1gt_7,282). happensAt(walking(fra1gt_7),282).
orientation(fra1gt_6,128,283). coords(fra1gt_6,202,71,283).  visible(fra1gt_6,283). happensAt(walking(fra1gt_6),283). orientation(fra1gt_7,137,283). coords(fra1gt_7,140,47,283).  visible(fra1gt_7,283). happensAt(walking(fra1gt_7),283).
orientation(fra1gt_6,128,284). coords(fra1gt_6,201,71,284).  visible(fra1gt_6,284). happensAt(walking(fra1gt_6),284). orientation(fra1gt_7,137,284). coords(fra1gt_7,139,47,284).  visible(fra1gt_7,284). happensAt(walking(fra1gt_7),284).
orientation(fra1gt_6,128,285). coords(fra1gt_6,200,72,285).  visible(fra1gt_6,285). happensAt(walking(fra1gt_6),285). orientation(fra1gt_7,142,285). coords(fra1gt_7,139,47,285).  visible(fra1gt_7,285). happensAt(walking(fra1gt_7),285).
orientation(fra1gt_6,128,286). coords(fra1gt_6,199,72,286).  visible(fra1gt_6,286). happensAt(walking(fra1gt_6),286). orientation(fra1gt_7,142,286). coords(fra1gt_7,137,47,286).  visible(fra1gt_7,286). happensAt(walking(fra1gt_7),286).
orientation(fra1gt_6,128,287). coords(fra1gt_6,198,72,287).  visible(fra1gt_6,287). happensAt(walking(fra1gt_6),287). orientation(fra1gt_7,142,287). coords(fra1gt_7,137,48,287).  visible(fra1gt_7,287). happensAt(walking(fra1gt_7),287).
orientation(fra1gt_6,128,288). coords(fra1gt_6,197,73,288).  visible(fra1gt_6,288). happensAt(walking(fra1gt_6),288). orientation(fra1gt_7,142,288). coords(fra1gt_7,137,49,288).  visible(fra1gt_7,288). happensAt(walking(fra1gt_7),288).
orientation(fra1gt_6,128,289). coords(fra1gt_6,195,73,289).  visible(fra1gt_6,289). happensAt(walking(fra1gt_6),289). orientation(fra1gt_7,142,289). coords(fra1gt_7,137,50,289).  visible(fra1gt_7,289). happensAt(walking(fra1gt_7),289).
orientation(fra1gt_6,128,290). coords(fra1gt_6,194,74,290).  visible(fra1gt_6,290). happensAt(walking(fra1gt_6),290). orientation(fra1gt_7,142,290). coords(fra1gt_7,137,51,290).  visible(fra1gt_7,290). happensAt(walking(fra1gt_7),290).
orientation(fra1gt_6,128,291). coords(fra1gt_6,191,74,291).  visible(fra1gt_6,291). happensAt(walking(fra1gt_6),291). orientation(fra1gt_7,142,291). coords(fra1gt_7,137,51,291).  visible(fra1gt_7,291). happensAt(walking(fra1gt_7),291).
orientation(fra1gt_6,128,292). coords(fra1gt_6,190,74,292).  visible(fra1gt_6,292). happensAt(walking(fra1gt_6),292). orientation(fra1gt_7,142,292). coords(fra1gt_7,137,51,292).  visible(fra1gt_7,292). happensAt(walking(fra1gt_7),292).
orientation(fra1gt_6,128,293). coords(fra1gt_6,188,74,293).  visible(fra1gt_6,293). happensAt(walking(fra1gt_6),293). orientation(fra1gt_7,142,293). coords(fra1gt_7,137,51,293).  visible(fra1gt_7,293). happensAt(walking(fra1gt_7),293).
orientation(fra1gt_6,128,294). coords(fra1gt_6,185,75,294).  visible(fra1gt_6,294). happensAt(walking(fra1gt_6),294). orientation(fra1gt_7,142,294). coords(fra1gt_7,137,51,294).  visible(fra1gt_7,294). happensAt(walking(fra1gt_7),294).
#show holdsAt/2.