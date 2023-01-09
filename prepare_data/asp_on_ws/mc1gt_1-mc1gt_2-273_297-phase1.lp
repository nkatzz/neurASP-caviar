
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

	orientation(mc1gt_1,165,273). coords(mc1gt_1,213,194,273).  visible(mc1gt_1,273). happensAt(walking(mc1gt_1),273). orientation(mc1gt_2,153,273). coords(mc1gt_2,169,159,273).  visible(mc1gt_2,273). happensAt(walking(mc1gt_2),273).
orientation(mc1gt_1,168,274). coords(mc1gt_1,214,196,274).  visible(mc1gt_1,274). happensAt(walking(mc1gt_1),274). orientation(mc1gt_2,152,274). coords(mc1gt_2,171,159,274).  visible(mc1gt_2,274). happensAt(walking(mc1gt_2),274).
orientation(mc1gt_1,171,275). coords(mc1gt_1,215,200,275).  visible(mc1gt_1,275). happensAt(walking(mc1gt_1),275). orientation(mc1gt_2,152,275). coords(mc1gt_2,172,161,275).  visible(mc1gt_2,275). happensAt(walking(mc1gt_2),275).
orientation(mc1gt_1,170,276). coords(mc1gt_1,217,204,276).  visible(mc1gt_1,276). happensAt(walking(mc1gt_1),276). orientation(mc1gt_2,160,276). coords(mc1gt_2,174,162,276).  visible(mc1gt_2,276). happensAt(walking(mc1gt_2),276).
orientation(mc1gt_1,169,277). coords(mc1gt_1,218,208,277).  visible(mc1gt_1,277). happensAt(walking(mc1gt_1),277). orientation(mc1gt_2,161,277). coords(mc1gt_2,176,162,277).  visible(mc1gt_2,277). happensAt(walking(mc1gt_2),277).
orientation(mc1gt_1,175,278). coords(mc1gt_1,218,210,278).  visible(mc1gt_1,278). happensAt(walking(mc1gt_1),278). orientation(mc1gt_2,164,278). coords(mc1gt_2,176,163,278).  visible(mc1gt_2,278). happensAt(walking(mc1gt_2),278).
orientation(mc1gt_1,175,279). coords(mc1gt_1,219,211,279).  visible(mc1gt_1,279). happensAt(walking(mc1gt_1),279). orientation(mc1gt_2,164,279). coords(mc1gt_2,177,164,279).  visible(mc1gt_2,279). happensAt(walking(mc1gt_2),279).
orientation(mc1gt_1,174,280). coords(mc1gt_1,219,211,280).  visible(mc1gt_1,280). happensAt(walking(mc1gt_1),280). orientation(mc1gt_2,152,280). coords(mc1gt_2,177,164,280).  visible(mc1gt_2,280). happensAt(walking(mc1gt_2),280).
orientation(mc1gt_1,178,281). coords(mc1gt_1,220,211,281).  visible(mc1gt_1,281). happensAt(walking(mc1gt_1),281). orientation(mc1gt_2,152,281). coords(mc1gt_2,178,164,281).  visible(mc1gt_2,281). happensAt(walking(mc1gt_2),281).
orientation(mc1gt_1,1,282). coords(mc1gt_1,221,211,282).  visible(mc1gt_1,282). happensAt(walking(mc1gt_1),282). orientation(mc1gt_2,152,282). coords(mc1gt_2,178,165,282).  visible(mc1gt_2,282). happensAt(walking(mc1gt_2),282).
orientation(mc1gt_1,6,283). coords(mc1gt_1,221,212,283).  visible(mc1gt_1,283). happensAt(walking(mc1gt_1),283). orientation(mc1gt_2,152,283). coords(mc1gt_2,179,166,283).  visible(mc1gt_2,283). happensAt(walking(mc1gt_2),283).
orientation(mc1gt_1,4,284). coords(mc1gt_1,222,213,284).  visible(mc1gt_1,284). happensAt(walking(mc1gt_1),284). orientation(mc1gt_2,152,284). coords(mc1gt_2,180,168,284).  visible(mc1gt_2,284). happensAt(walking(mc1gt_2),284).
orientation(mc1gt_1,4,285). coords(mc1gt_1,222,215,285).  visible(mc1gt_1,285). happensAt(walking(mc1gt_1),285). orientation(mc1gt_2,152,285). coords(mc1gt_2,180,170,285).  visible(mc1gt_2,285). happensAt(walking(mc1gt_2),285).
orientation(mc1gt_1,7,286). coords(mc1gt_1,222,218,286).  visible(mc1gt_1,286). happensAt(walking(mc1gt_1),286). orientation(mc1gt_2,156,286). coords(mc1gt_2,183,173,286).  visible(mc1gt_2,286). happensAt(walking(mc1gt_2),286).
orientation(mc1gt_1,3,287). coords(mc1gt_1,222,221,287).  visible(mc1gt_1,287). happensAt(walking(mc1gt_1),287). orientation(mc1gt_2,165,287). coords(mc1gt_2,185,176,287).  visible(mc1gt_2,287). happensAt(walking(mc1gt_2),287).
orientation(mc1gt_1,1,288). coords(mc1gt_1,222,225,288).  visible(mc1gt_1,288). happensAt(walking(mc1gt_1),288). orientation(mc1gt_2,163,288). coords(mc1gt_2,187,178,288).  visible(mc1gt_2,288). happensAt(walking(mc1gt_2),288).
orientation(mc1gt_1,3,289). coords(mc1gt_1,222,228,289).  visible(mc1gt_1,289). happensAt(walking(mc1gt_1),289). orientation(mc1gt_2,163,289). coords(mc1gt_2,188,179,289).  visible(mc1gt_2,289). happensAt(walking(mc1gt_2),289).
orientation(mc1gt_1,0,290). coords(mc1gt_1,222,230,290).  visible(mc1gt_1,290). happensAt(walking(mc1gt_1),290). orientation(mc1gt_2,163,290). coords(mc1gt_2,190,180,290).  visible(mc1gt_2,290). happensAt(walking(mc1gt_2),290).
orientation(mc1gt_1,171,291). coords(mc1gt_1,222,232,291).  visible(mc1gt_1,291). happensAt(walking(mc1gt_1),291). orientation(mc1gt_2,166,291). coords(mc1gt_2,191,182,291).  visible(mc1gt_2,291). happensAt(walking(mc1gt_2),291).
orientation(mc1gt_1,172,292). coords(mc1gt_1,222,235,292).  visible(mc1gt_1,292). happensAt(walking(mc1gt_1),292). orientation(mc1gt_2,159,292). coords(mc1gt_2,191,182,292).  visible(mc1gt_2,292). happensAt(walking(mc1gt_2),292).
orientation(mc1gt_1,174,293). coords(mc1gt_1,222,237,293).  visible(mc1gt_1,293). happensAt(walking(mc1gt_1),293). orientation(mc1gt_2,160,293). coords(mc1gt_2,192,183,293).  visible(mc1gt_2,293). happensAt(walking(mc1gt_2),293).
orientation(mc1gt_1,0,294). coords(mc1gt_1,222,239,294).  visible(mc1gt_1,294). happensAt(walking(mc1gt_1),294). orientation(mc1gt_2,165,294). coords(mc1gt_2,192,184,294).  visible(mc1gt_2,294). happensAt(walking(mc1gt_2),294).
orientation(mc1gt_1,3,295). coords(mc1gt_1,222,239,295).  visible(mc1gt_1,295). happensAt(walking(mc1gt_1),295). orientation(mc1gt_2,167,295). coords(mc1gt_2,192,184,295).  visible(mc1gt_2,295). happensAt(walking(mc1gt_2),295).
orientation(mc1gt_1,178,296). coords(mc1gt_1,222,240,296).  visible(mc1gt_1,296). happensAt(walking(mc1gt_1),296). orientation(mc1gt_2,172,296). coords(mc1gt_2,193,185,296).  visible(mc1gt_2,296). happensAt(walking(mc1gt_2),296).
orientation(mc1gt_1,3,297). coords(mc1gt_1,222,240,297).  visible(mc1gt_1,297). happensAt(walking(mc1gt_1),297). orientation(mc1gt_2,168,297). coords(mc1gt_2,194,187,297).  visible(mc1gt_2,297). happensAt(walking(mc1gt_2),297).
#show holdsAt/2.