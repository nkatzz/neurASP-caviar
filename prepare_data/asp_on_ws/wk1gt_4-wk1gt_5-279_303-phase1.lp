
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

	orientation(wk1gt_4,155,279). coords(wk1gt_4,51,177,279).  visible(wk1gt_4,279). happensAt(active(wk1gt_4),279). orientation(wk1gt_5,162,279). coords(wk1gt_5,50,166,279).  visible(wk1gt_5,279). happensAt(active(wk1gt_5),279).
orientation(wk1gt_4,156,280). coords(wk1gt_4,51,177,280).  visible(wk1gt_4,280). happensAt(active(wk1gt_4),280). orientation(wk1gt_5,162,280). coords(wk1gt_5,50,166,280).  visible(wk1gt_5,280). happensAt(active(wk1gt_5),280).
orientation(wk1gt_4,157,281). coords(wk1gt_4,51,176,281).  visible(wk1gt_4,281). happensAt(active(wk1gt_4),281). orientation(wk1gt_5,162,281). coords(wk1gt_5,50,166,281).  visible(wk1gt_5,281). happensAt(active(wk1gt_5),281).
orientation(wk1gt_4,161,282). coords(wk1gt_4,51,176,282).  visible(wk1gt_4,282). happensAt(active(wk1gt_4),282). orientation(wk1gt_5,162,282). coords(wk1gt_5,50,166,282).  visible(wk1gt_5,282). happensAt(active(wk1gt_5),282).
orientation(wk1gt_4,162,283). coords(wk1gt_4,51,176,283).  visible(wk1gt_4,283). happensAt(active(wk1gt_4),283). orientation(wk1gt_5,162,283). coords(wk1gt_5,50,166,283).  visible(wk1gt_5,283). happensAt(active(wk1gt_5),283).
orientation(wk1gt_4,159,284). coords(wk1gt_4,51,176,284).  visible(wk1gt_4,284). happensAt(active(wk1gt_4),284). orientation(wk1gt_5,162,284). coords(wk1gt_5,50,166,284).  visible(wk1gt_5,284). happensAt(active(wk1gt_5),284).
orientation(wk1gt_4,160,285). coords(wk1gt_4,51,176,285).  visible(wk1gt_4,285). happensAt(active(wk1gt_4),285). orientation(wk1gt_5,162,285). coords(wk1gt_5,50,166,285).  visible(wk1gt_5,285). happensAt(active(wk1gt_5),285).
orientation(wk1gt_4,160,286). coords(wk1gt_4,51,177,286).  visible(wk1gt_4,286). happensAt(active(wk1gt_4),286). orientation(wk1gt_5,162,286). coords(wk1gt_5,50,166,286).  visible(wk1gt_5,286). happensAt(active(wk1gt_5),286).
orientation(wk1gt_4,160,287). coords(wk1gt_4,51,177,287).  visible(wk1gt_4,287). happensAt(active(wk1gt_4),287). orientation(wk1gt_5,162,287). coords(wk1gt_5,50,166,287).  visible(wk1gt_5,287). happensAt(active(wk1gt_5),287).
orientation(wk1gt_4,158,288). coords(wk1gt_4,52,177,288).  visible(wk1gt_4,288). happensAt(active(wk1gt_4),288). orientation(wk1gt_5,162,288). coords(wk1gt_5,50,166,288).  visible(wk1gt_5,288). happensAt(active(wk1gt_5),288).
orientation(wk1gt_4,166,289). coords(wk1gt_4,52,177,289).  visible(wk1gt_4,289). happensAt(active(wk1gt_4),289). orientation(wk1gt_5,162,289). coords(wk1gt_5,50,166,289).  visible(wk1gt_5,289). happensAt(active(wk1gt_5),289).
orientation(wk1gt_4,160,290). coords(wk1gt_4,52,177,290).  visible(wk1gt_4,290). happensAt(active(wk1gt_4),290). orientation(wk1gt_5,162,290). coords(wk1gt_5,50,165,290).  visible(wk1gt_5,290). happensAt(active(wk1gt_5),290).
orientation(wk1gt_4,157,291). coords(wk1gt_4,52,177,291).  visible(wk1gt_4,291). happensAt(active(wk1gt_4),291). orientation(wk1gt_5,162,291). coords(wk1gt_5,50,165,291).  visible(wk1gt_5,291). happensAt(active(wk1gt_5),291).
orientation(wk1gt_4,162,292). coords(wk1gt_4,52,177,292).  visible(wk1gt_4,292). happensAt(active(wk1gt_4),292). orientation(wk1gt_5,162,292). coords(wk1gt_5,50,165,292).  visible(wk1gt_5,292). happensAt(active(wk1gt_5),292).
orientation(wk1gt_4,161,293). coords(wk1gt_4,52,177,293).  visible(wk1gt_4,293). happensAt(active(wk1gt_4),293). orientation(wk1gt_5,162,293). coords(wk1gt_5,51,165,293).  visible(wk1gt_5,293). happensAt(active(wk1gt_5),293).
orientation(wk1gt_4,157,294). coords(wk1gt_4,52,177,294).  visible(wk1gt_4,294). happensAt(active(wk1gt_4),294). orientation(wk1gt_5,162,294). coords(wk1gt_5,51,165,294).  visible(wk1gt_5,294). happensAt(active(wk1gt_5),294).
orientation(wk1gt_4,158,295). coords(wk1gt_4,52,177,295).  visible(wk1gt_4,295). happensAt(active(wk1gt_4),295). orientation(wk1gt_5,162,295). coords(wk1gt_5,50,164,295).  visible(wk1gt_5,295). happensAt(active(wk1gt_5),295).
orientation(wk1gt_4,161,296). coords(wk1gt_4,52,176,296).  visible(wk1gt_4,296). happensAt(active(wk1gt_4),296). orientation(wk1gt_5,162,296). coords(wk1gt_5,50,164,296).  visible(wk1gt_5,296). happensAt(active(wk1gt_5),296).
orientation(wk1gt_4,160,297). coords(wk1gt_4,52,176,297).  visible(wk1gt_4,297). happensAt(active(wk1gt_4),297). orientation(wk1gt_5,162,297). coords(wk1gt_5,50,164,297).  visible(wk1gt_5,297). happensAt(active(wk1gt_5),297).
orientation(wk1gt_4,161,298). coords(wk1gt_4,52,175,298).  visible(wk1gt_4,298). happensAt(active(wk1gt_4),298). orientation(wk1gt_5,162,298). coords(wk1gt_5,50,164,298).  visible(wk1gt_5,298). happensAt(active(wk1gt_5),298).
orientation(wk1gt_4,167,299). coords(wk1gt_4,52,175,299).  visible(wk1gt_4,299). happensAt(active(wk1gt_4),299). orientation(wk1gt_5,162,299). coords(wk1gt_5,50,164,299).  visible(wk1gt_5,299). happensAt(active(wk1gt_5),299).
orientation(wk1gt_4,168,300). coords(wk1gt_4,52,174,300).  visible(wk1gt_4,300). happensAt(active(wk1gt_4),300). orientation(wk1gt_5,162,300). coords(wk1gt_5,50,164,300).  visible(wk1gt_5,300). happensAt(active(wk1gt_5),300).
orientation(wk1gt_4,173,301). coords(wk1gt_4,52,174,301).  visible(wk1gt_4,301). happensAt(active(wk1gt_4),301). orientation(wk1gt_5,162,301). coords(wk1gt_5,50,164,301).  visible(wk1gt_5,301). happensAt(active(wk1gt_5),301).
orientation(wk1gt_4,170,302). coords(wk1gt_4,52,173,302).  visible(wk1gt_4,302). happensAt(active(wk1gt_4),302). orientation(wk1gt_5,162,302). coords(wk1gt_5,51,164,302).  visible(wk1gt_5,302). happensAt(active(wk1gt_5),302).
orientation(wk1gt_4,170,303). coords(wk1gt_4,52,173,303).  visible(wk1gt_4,303). happensAt(active(wk1gt_4),303). orientation(wk1gt_5,162,303). coords(wk1gt_5,50,164,303).  visible(wk1gt_5,303). happensAt(active(wk1gt_5),303).
#show holdsAt/2.