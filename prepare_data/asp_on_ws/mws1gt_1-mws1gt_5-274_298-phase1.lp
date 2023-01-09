
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

	orientation(mws1gt_1,155,274). coords(mws1gt_1,95,109,274).  visible(mws1gt_1,274). happensAt(walking(mws1gt_1),274). orientation(mws1gt_5,161,274). coords(mws1gt_5,40,156,274).  visible(mws1gt_5,274). happensAt(walking(mws1gt_5),274).
orientation(mws1gt_1,155,275). coords(mws1gt_1,95,109,275).  visible(mws1gt_1,275). happensAt(walking(mws1gt_1),275). orientation(mws1gt_5,161,275). coords(mws1gt_5,40,157,275).  visible(mws1gt_5,275). happensAt(walking(mws1gt_5),275).
orientation(mws1gt_1,155,276). coords(mws1gt_1,93,109,276).  visible(mws1gt_1,276). happensAt(walking(mws1gt_1),276). orientation(mws1gt_5,161,276). coords(mws1gt_5,39,158,276).  visible(mws1gt_5,276). happensAt(walking(mws1gt_5),276).
orientation(mws1gt_1,155,277). coords(mws1gt_1,93,109,277).  visible(mws1gt_1,277). happensAt(walking(mws1gt_1),277). orientation(mws1gt_5,161,277). coords(mws1gt_5,39,159,277).  visible(mws1gt_5,277). happensAt(walking(mws1gt_5),277).
orientation(mws1gt_1,155,278). coords(mws1gt_1,93,109,278).  visible(mws1gt_1,278). happensAt(walking(mws1gt_1),278). orientation(mws1gt_5,161,278). coords(mws1gt_5,38,160,278).  visible(mws1gt_5,278). happensAt(walking(mws1gt_5),278).
orientation(mws1gt_1,155,279). coords(mws1gt_1,93,109,279).  visible(mws1gt_1,279). happensAt(walking(mws1gt_1),279). orientation(mws1gt_5,161,279). coords(mws1gt_5,38,162,279).  visible(mws1gt_5,279). happensAt(walking(mws1gt_5),279).
orientation(mws1gt_1,155,280). coords(mws1gt_1,93,108,280).  visible(mws1gt_1,280). happensAt(walking(mws1gt_1),280). orientation(mws1gt_5,161,280). coords(mws1gt_5,38,162,280).  visible(mws1gt_5,280). happensAt(walking(mws1gt_5),280).
orientation(mws1gt_1,155,281). coords(mws1gt_1,93,108,281).  visible(mws1gt_1,281). happensAt(walking(mws1gt_1),281). orientation(mws1gt_5,161,281). coords(mws1gt_5,38,163,281).  visible(mws1gt_5,281). happensAt(walking(mws1gt_5),281).
orientation(mws1gt_1,155,282). coords(mws1gt_1,92,108,282).  visible(mws1gt_1,282). happensAt(walking(mws1gt_1),282). orientation(mws1gt_5,161,282). coords(mws1gt_5,38,164,282).  visible(mws1gt_5,282). happensAt(walking(mws1gt_5),282).
orientation(mws1gt_1,155,283). coords(mws1gt_1,92,108,283).  visible(mws1gt_1,283). happensAt(walking(mws1gt_1),283). orientation(mws1gt_5,161,283). coords(mws1gt_5,38,164,283).  visible(mws1gt_5,283). happensAt(walking(mws1gt_5),283).
orientation(mws1gt_1,155,284). coords(mws1gt_1,92,107,284).  visible(mws1gt_1,284). happensAt(walking(mws1gt_1),284). orientation(mws1gt_5,161,284). coords(mws1gt_5,37,164,284).  visible(mws1gt_5,284). happensAt(walking(mws1gt_5),284).
orientation(mws1gt_1,155,285). coords(mws1gt_1,91,107,285).  visible(mws1gt_1,285). happensAt(walking(mws1gt_1),285). orientation(mws1gt_5,161,285). coords(mws1gt_5,37,165,285).  visible(mws1gt_5,285). happensAt(walking(mws1gt_5),285).
orientation(mws1gt_1,155,286). coords(mws1gt_1,91,106,286).  visible(mws1gt_1,286). happensAt(walking(mws1gt_1),286). orientation(mws1gt_5,161,286). coords(mws1gt_5,37,165,286).  visible(mws1gt_5,286). happensAt(walking(mws1gt_5),286).
orientation(mws1gt_1,155,287). coords(mws1gt_1,89,106,287).  visible(mws1gt_1,287). happensAt(walking(mws1gt_1),287). orientation(mws1gt_5,161,287). coords(mws1gt_5,37,166,287).  visible(mws1gt_5,287). happensAt(walking(mws1gt_5),287).
orientation(mws1gt_1,155,288). coords(mws1gt_1,88,105,288).  visible(mws1gt_1,288). happensAt(walking(mws1gt_1),288). orientation(mws1gt_5,161,288). coords(mws1gt_5,37,166,288).  visible(mws1gt_5,288). happensAt(walking(mws1gt_5),288).
orientation(mws1gt_1,155,289). coords(mws1gt_1,88,105,289).  visible(mws1gt_1,289). happensAt(walking(mws1gt_1),289). orientation(mws1gt_5,161,289). coords(mws1gt_5,37,167,289).  visible(mws1gt_5,289). happensAt(walking(mws1gt_5),289).
orientation(mws1gt_1,155,290). coords(mws1gt_1,87,104,290).  visible(mws1gt_1,290). happensAt(walking(mws1gt_1),290). orientation(mws1gt_5,161,290). coords(mws1gt_5,37,168,290).  visible(mws1gt_5,290). happensAt(walking(mws1gt_5),290).
orientation(mws1gt_1,155,291). coords(mws1gt_1,87,104,291).  visible(mws1gt_1,291). happensAt(walking(mws1gt_1),291). orientation(mws1gt_5,168,291). coords(mws1gt_5,37,169,291).  visible(mws1gt_5,291). happensAt(walking(mws1gt_5),291).
orientation(mws1gt_1,155,292). coords(mws1gt_1,87,104,292).  visible(mws1gt_1,292). happensAt(walking(mws1gt_1),292). orientation(mws1gt_5,168,292). coords(mws1gt_5,37,170,292).  visible(mws1gt_5,292). happensAt(walking(mws1gt_5),292).
orientation(mws1gt_1,155,293). coords(mws1gt_1,87,104,293).  visible(mws1gt_1,293). happensAt(walking(mws1gt_1),293). orientation(mws1gt_5,168,293). coords(mws1gt_5,37,171,293).  visible(mws1gt_5,293). happensAt(walking(mws1gt_5),293).
orientation(mws1gt_1,155,294). coords(mws1gt_1,87,103,294).  visible(mws1gt_1,294). happensAt(walking(mws1gt_1),294). orientation(mws1gt_5,168,294). coords(mws1gt_5,37,172,294).  visible(mws1gt_5,294). happensAt(walking(mws1gt_5),294).
orientation(mws1gt_1,155,295). coords(mws1gt_1,87,103,295).  visible(mws1gt_1,295). happensAt(walking(mws1gt_1),295). orientation(mws1gt_5,168,295). coords(mws1gt_5,37,174,295).  visible(mws1gt_5,295). happensAt(walking(mws1gt_5),295).
orientation(mws1gt_1,155,296). coords(mws1gt_1,87,103,296).  visible(mws1gt_1,296). happensAt(walking(mws1gt_1),296). orientation(mws1gt_5,168,296). coords(mws1gt_5,36,175,296).  visible(mws1gt_5,296). happensAt(walking(mws1gt_5),296).
orientation(mws1gt_1,155,297). coords(mws1gt_1,86,103,297).  visible(mws1gt_1,297). happensAt(walking(mws1gt_1),297). orientation(mws1gt_5,168,297). coords(mws1gt_5,36,176,297).  visible(mws1gt_5,297). happensAt(walking(mws1gt_5),297).
orientation(mws1gt_1,155,298). coords(mws1gt_1,86,103,298).  visible(mws1gt_1,298). happensAt(walking(mws1gt_1),298). orientation(mws1gt_5,168,298). coords(mws1gt_5,36,177,298).  visible(mws1gt_5,298). happensAt(walking(mws1gt_5),298).
#show holdsAt/2.