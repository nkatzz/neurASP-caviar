
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

	orientation(wk1gt_1,3,272). coords(wk1gt_1,213,241,272).  visible(wk1gt_1,272). happensAt(walking(wk1gt_1),272). orientation(wk1gt_4,153,272). coords(wk1gt_4,49,178,272).  visible(wk1gt_4,272). happensAt(active(wk1gt_4),272).
orientation(wk1gt_1,3,273). coords(wk1gt_1,212,239,273).  visible(wk1gt_1,273). happensAt(walking(wk1gt_1),273). orientation(wk1gt_4,157,273). coords(wk1gt_4,49,178,273).  visible(wk1gt_4,273). happensAt(active(wk1gt_4),273).
orientation(wk1gt_1,3,274). coords(wk1gt_1,212,236,274).  visible(wk1gt_1,274). happensAt(walking(wk1gt_1),274). orientation(wk1gt_4,156,274). coords(wk1gt_4,49,178,274).  visible(wk1gt_4,274). happensAt(active(wk1gt_4),274).
orientation(wk1gt_1,3,275). coords(wk1gt_1,211,232,275).  visible(wk1gt_1,275). happensAt(walking(wk1gt_1),275). orientation(wk1gt_4,153,275). coords(wk1gt_4,49,178,275).  visible(wk1gt_4,275). happensAt(active(wk1gt_4),275).
orientation(wk1gt_1,9,276). coords(wk1gt_1,211,230,276).  visible(wk1gt_1,276). happensAt(walking(wk1gt_1),276). orientation(wk1gt_4,154,276). coords(wk1gt_4,49,177,276).  visible(wk1gt_4,276). happensAt(active(wk1gt_4),276).
orientation(wk1gt_1,9,277). coords(wk1gt_1,210,229,277).  visible(wk1gt_1,277). happensAt(walking(wk1gt_1),277). orientation(wk1gt_4,154,277). coords(wk1gt_4,49,177,277).  visible(wk1gt_4,277). happensAt(active(wk1gt_4),277).
orientation(wk1gt_1,174,278). coords(wk1gt_1,210,229,278).  visible(wk1gt_1,278). happensAt(walking(wk1gt_1),278). orientation(wk1gt_4,156,278). coords(wk1gt_4,50,177,278).  visible(wk1gt_4,278). happensAt(active(wk1gt_4),278).
orientation(wk1gt_1,167,279). coords(wk1gt_1,210,229,279).  visible(wk1gt_1,279). happensAt(walking(wk1gt_1),279). orientation(wk1gt_4,155,279). coords(wk1gt_4,51,177,279).  visible(wk1gt_4,279). happensAt(active(wk1gt_4),279).
orientation(wk1gt_1,167,280). coords(wk1gt_1,209,228,280).  visible(wk1gt_1,280). happensAt(walking(wk1gt_1),280). orientation(wk1gt_4,156,280). coords(wk1gt_4,51,177,280).  visible(wk1gt_4,280). happensAt(active(wk1gt_4),280).
orientation(wk1gt_1,155,281). coords(wk1gt_1,207,227,281).  visible(wk1gt_1,281). happensAt(walking(wk1gt_1),281). orientation(wk1gt_4,157,281). coords(wk1gt_4,51,176,281).  visible(wk1gt_4,281). happensAt(active(wk1gt_4),281).
orientation(wk1gt_1,155,282). coords(wk1gt_1,206,224,282).  visible(wk1gt_1,282). happensAt(walking(wk1gt_1),282). orientation(wk1gt_4,161,282). coords(wk1gt_4,51,176,282).  visible(wk1gt_4,282). happensAt(active(wk1gt_4),282).
orientation(wk1gt_1,155,283). coords(wk1gt_1,203,222,283).  visible(wk1gt_1,283). happensAt(walking(wk1gt_1),283). orientation(wk1gt_4,162,283). coords(wk1gt_4,51,176,283).  visible(wk1gt_4,283). happensAt(active(wk1gt_4),283).
orientation(wk1gt_1,155,284). coords(wk1gt_1,203,218,284).  visible(wk1gt_1,284). happensAt(walking(wk1gt_1),284). orientation(wk1gt_4,159,284). coords(wk1gt_4,51,176,284).  visible(wk1gt_4,284). happensAt(active(wk1gt_4),284).
orientation(wk1gt_1,165,285). coords(wk1gt_1,202,216,285).  visible(wk1gt_1,285). happensAt(walking(wk1gt_1),285). orientation(wk1gt_4,160,285). coords(wk1gt_4,51,176,285).  visible(wk1gt_4,285). happensAt(active(wk1gt_4),285).
orientation(wk1gt_1,165,286). coords(wk1gt_1,202,214,286).  visible(wk1gt_1,286). happensAt(walking(wk1gt_1),286). orientation(wk1gt_4,160,286). coords(wk1gt_4,51,177,286).  visible(wk1gt_4,286). happensAt(active(wk1gt_4),286).
orientation(wk1gt_1,165,287). coords(wk1gt_1,201,212,287).  visible(wk1gt_1,287). happensAt(walking(wk1gt_1),287). orientation(wk1gt_4,160,287). coords(wk1gt_4,51,177,287).  visible(wk1gt_4,287). happensAt(active(wk1gt_4),287).
orientation(wk1gt_1,165,288). coords(wk1gt_1,201,210,288).  visible(wk1gt_1,288). happensAt(walking(wk1gt_1),288). orientation(wk1gt_4,158,288). coords(wk1gt_4,52,177,288).  visible(wk1gt_4,288). happensAt(active(wk1gt_4),288).
orientation(wk1gt_1,165,289). coords(wk1gt_1,200,209,289).  visible(wk1gt_1,289). happensAt(walking(wk1gt_1),289). orientation(wk1gt_4,166,289). coords(wk1gt_4,52,177,289).  visible(wk1gt_4,289). happensAt(active(wk1gt_4),289).
orientation(wk1gt_1,160,290). coords(wk1gt_1,200,208,290).  visible(wk1gt_1,290). happensAt(walking(wk1gt_1),290). orientation(wk1gt_4,160,290). coords(wk1gt_4,52,177,290).  visible(wk1gt_4,290). happensAt(active(wk1gt_4),290).
orientation(wk1gt_1,154,291). coords(wk1gt_1,200,207,291).  visible(wk1gt_1,291). happensAt(walking(wk1gt_1),291). orientation(wk1gt_4,157,291). coords(wk1gt_4,52,177,291).  visible(wk1gt_4,291). happensAt(active(wk1gt_4),291).
orientation(wk1gt_1,154,292). coords(wk1gt_1,198,206,292).  visible(wk1gt_1,292). happensAt(walking(wk1gt_1),292). orientation(wk1gt_4,162,292). coords(wk1gt_4,52,177,292).  visible(wk1gt_4,292). happensAt(active(wk1gt_4),292).
orientation(wk1gt_1,154,293). coords(wk1gt_1,198,204,293).  visible(wk1gt_1,293). happensAt(walking(wk1gt_1),293). orientation(wk1gt_4,161,293). coords(wk1gt_4,52,177,293).  visible(wk1gt_4,293). happensAt(active(wk1gt_4),293).
orientation(wk1gt_1,154,294). coords(wk1gt_1,196,202,294).  visible(wk1gt_1,294). happensAt(walking(wk1gt_1),294). orientation(wk1gt_4,157,294). coords(wk1gt_4,52,177,294).  visible(wk1gt_4,294). happensAt(active(wk1gt_4),294).
orientation(wk1gt_1,163,295). coords(wk1gt_1,193,200,295).  visible(wk1gt_1,295). happensAt(walking(wk1gt_1),295). orientation(wk1gt_4,158,295). coords(wk1gt_4,52,177,295).  visible(wk1gt_4,295). happensAt(active(wk1gt_4),295).
orientation(wk1gt_1,163,296). coords(wk1gt_1,193,198,296).  visible(wk1gt_1,296). happensAt(walking(wk1gt_1),296). orientation(wk1gt_4,161,296). coords(wk1gt_4,52,176,296).  visible(wk1gt_4,296). happensAt(active(wk1gt_4),296).
#show holdsAt/2.