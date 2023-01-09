
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

	orientation(wk1gt_1,3,272). coords(wk1gt_1,213,241,272).  visible(wk1gt_1,272). happensAt(walking(wk1gt_1),272). orientation(wk1gt_5,162,272). coords(wk1gt_5,50,166,272).  visible(wk1gt_5,272). happensAt(active(wk1gt_5),272).
orientation(wk1gt_1,3,273). coords(wk1gt_1,212,239,273).  visible(wk1gt_1,273). happensAt(walking(wk1gt_1),273). orientation(wk1gt_5,162,273). coords(wk1gt_5,50,166,273).  visible(wk1gt_5,273). happensAt(active(wk1gt_5),273).
orientation(wk1gt_1,3,274). coords(wk1gt_1,212,236,274).  visible(wk1gt_1,274). happensAt(walking(wk1gt_1),274). orientation(wk1gt_5,162,274). coords(wk1gt_5,50,166,274).  visible(wk1gt_5,274). happensAt(active(wk1gt_5),274).
orientation(wk1gt_1,3,275). coords(wk1gt_1,211,232,275).  visible(wk1gt_1,275). happensAt(walking(wk1gt_1),275). orientation(wk1gt_5,162,275). coords(wk1gt_5,50,166,275).  visible(wk1gt_5,275). happensAt(active(wk1gt_5),275).
orientation(wk1gt_1,9,276). coords(wk1gt_1,211,230,276).  visible(wk1gt_1,276). happensAt(walking(wk1gt_1),276). orientation(wk1gt_5,162,276). coords(wk1gt_5,50,166,276).  visible(wk1gt_5,276). happensAt(active(wk1gt_5),276).
orientation(wk1gt_1,9,277). coords(wk1gt_1,210,229,277).  visible(wk1gt_1,277). happensAt(walking(wk1gt_1),277). orientation(wk1gt_5,162,277). coords(wk1gt_5,50,166,277).  visible(wk1gt_5,277). happensAt(active(wk1gt_5),277).
orientation(wk1gt_1,174,278). coords(wk1gt_1,210,229,278).  visible(wk1gt_1,278). happensAt(walking(wk1gt_1),278). orientation(wk1gt_5,162,278). coords(wk1gt_5,50,166,278).  visible(wk1gt_5,278). happensAt(active(wk1gt_5),278).
orientation(wk1gt_1,167,279). coords(wk1gt_1,210,229,279).  visible(wk1gt_1,279). happensAt(walking(wk1gt_1),279). orientation(wk1gt_5,162,279). coords(wk1gt_5,50,166,279).  visible(wk1gt_5,279). happensAt(active(wk1gt_5),279).
orientation(wk1gt_1,167,280). coords(wk1gt_1,209,228,280).  visible(wk1gt_1,280). happensAt(walking(wk1gt_1),280). orientation(wk1gt_5,162,280). coords(wk1gt_5,50,166,280).  visible(wk1gt_5,280). happensAt(active(wk1gt_5),280).
orientation(wk1gt_1,155,281). coords(wk1gt_1,207,227,281).  visible(wk1gt_1,281). happensAt(walking(wk1gt_1),281). orientation(wk1gt_5,162,281). coords(wk1gt_5,50,166,281).  visible(wk1gt_5,281). happensAt(active(wk1gt_5),281).
orientation(wk1gt_1,155,282). coords(wk1gt_1,206,224,282).  visible(wk1gt_1,282). happensAt(walking(wk1gt_1),282). orientation(wk1gt_5,162,282). coords(wk1gt_5,50,166,282).  visible(wk1gt_5,282). happensAt(active(wk1gt_5),282).
orientation(wk1gt_1,155,283). coords(wk1gt_1,203,222,283).  visible(wk1gt_1,283). happensAt(walking(wk1gt_1),283). orientation(wk1gt_5,162,283). coords(wk1gt_5,50,166,283).  visible(wk1gt_5,283). happensAt(active(wk1gt_5),283).
orientation(wk1gt_1,155,284). coords(wk1gt_1,203,218,284).  visible(wk1gt_1,284). happensAt(walking(wk1gt_1),284). orientation(wk1gt_5,162,284). coords(wk1gt_5,50,166,284).  visible(wk1gt_5,284). happensAt(active(wk1gt_5),284).
orientation(wk1gt_1,165,285). coords(wk1gt_1,202,216,285).  visible(wk1gt_1,285). happensAt(walking(wk1gt_1),285). orientation(wk1gt_5,162,285). coords(wk1gt_5,50,166,285).  visible(wk1gt_5,285). happensAt(active(wk1gt_5),285).
orientation(wk1gt_1,165,286). coords(wk1gt_1,202,214,286).  visible(wk1gt_1,286). happensAt(walking(wk1gt_1),286). orientation(wk1gt_5,162,286). coords(wk1gt_5,50,166,286).  visible(wk1gt_5,286). happensAt(active(wk1gt_5),286).
orientation(wk1gt_1,165,287). coords(wk1gt_1,201,212,287).  visible(wk1gt_1,287). happensAt(walking(wk1gt_1),287). orientation(wk1gt_5,162,287). coords(wk1gt_5,50,166,287).  visible(wk1gt_5,287). happensAt(active(wk1gt_5),287).
orientation(wk1gt_1,165,288). coords(wk1gt_1,201,210,288).  visible(wk1gt_1,288). happensAt(walking(wk1gt_1),288). orientation(wk1gt_5,162,288). coords(wk1gt_5,50,166,288).  visible(wk1gt_5,288). happensAt(active(wk1gt_5),288).
orientation(wk1gt_1,165,289). coords(wk1gt_1,200,209,289).  visible(wk1gt_1,289). happensAt(walking(wk1gt_1),289). orientation(wk1gt_5,162,289). coords(wk1gt_5,50,166,289).  visible(wk1gt_5,289). happensAt(active(wk1gt_5),289).
orientation(wk1gt_1,160,290). coords(wk1gt_1,200,208,290).  visible(wk1gt_1,290). happensAt(walking(wk1gt_1),290). orientation(wk1gt_5,162,290). coords(wk1gt_5,50,165,290).  visible(wk1gt_5,290). happensAt(active(wk1gt_5),290).
orientation(wk1gt_1,154,291). coords(wk1gt_1,200,207,291).  visible(wk1gt_1,291). happensAt(walking(wk1gt_1),291). orientation(wk1gt_5,162,291). coords(wk1gt_5,50,165,291).  visible(wk1gt_5,291). happensAt(active(wk1gt_5),291).
orientation(wk1gt_1,154,292). coords(wk1gt_1,198,206,292).  visible(wk1gt_1,292). happensAt(walking(wk1gt_1),292). orientation(wk1gt_5,162,292). coords(wk1gt_5,50,165,292).  visible(wk1gt_5,292). happensAt(active(wk1gt_5),292).
orientation(wk1gt_1,154,293). coords(wk1gt_1,198,204,293).  visible(wk1gt_1,293). happensAt(walking(wk1gt_1),293). orientation(wk1gt_5,162,293). coords(wk1gt_5,51,165,293).  visible(wk1gt_5,293). happensAt(active(wk1gt_5),293).
orientation(wk1gt_1,154,294). coords(wk1gt_1,196,202,294).  visible(wk1gt_1,294). happensAt(walking(wk1gt_1),294). orientation(wk1gt_5,162,294). coords(wk1gt_5,51,165,294).  visible(wk1gt_5,294). happensAt(active(wk1gt_5),294).
orientation(wk1gt_1,163,295). coords(wk1gt_1,193,200,295).  visible(wk1gt_1,295). happensAt(walking(wk1gt_1),295). orientation(wk1gt_5,162,295). coords(wk1gt_5,50,164,295).  visible(wk1gt_5,295). happensAt(active(wk1gt_5),295).
orientation(wk1gt_1,163,296). coords(wk1gt_1,193,198,296).  visible(wk1gt_1,296). happensAt(walking(wk1gt_1),296). orientation(wk1gt_5,162,296). coords(wk1gt_5,50,164,296).  visible(wk1gt_5,296). happensAt(active(wk1gt_5),296).
#show holdsAt/2.