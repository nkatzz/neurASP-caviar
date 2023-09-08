
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

	orientation(bww1gt_1,166,278). coords(bww1gt_1,95,256,278).  visible(bww1gt_1,278). happensAt(walking(bww1gt_1),278). orientation(bww1gt_2,167,278). coords(bww1gt_2,50,165,278).  visible(bww1gt_2,278). happensAt(walking(bww1gt_2),278).
orientation(bww1gt_1,173,279). coords(bww1gt_1,95,256,279).  visible(bww1gt_1,279). happensAt(walking(bww1gt_1),279). orientation(bww1gt_2,167,279). coords(bww1gt_2,51,165,279).  visible(bww1gt_2,279). happensAt(walking(bww1gt_2),279).
orientation(bww1gt_1,173,280). coords(bww1gt_1,95,257,280).  visible(bww1gt_1,280). happensAt(walking(bww1gt_1),280). orientation(bww1gt_2,167,280). coords(bww1gt_2,51,165,280).  visible(bww1gt_2,280). happensAt(walking(bww1gt_2),280).
orientation(bww1gt_1,173,281). coords(bww1gt_1,95,257,281).  visible(bww1gt_1,281). happensAt(walking(bww1gt_1),281). orientation(bww1gt_2,167,281). coords(bww1gt_2,51,165,281).  visible(bww1gt_2,281). happensAt(walking(bww1gt_2),281).
orientation(bww1gt_1,173,282). coords(bww1gt_1,95,257,282).  visible(bww1gt_1,282). happensAt(walking(bww1gt_1),282). orientation(bww1gt_2,167,282). coords(bww1gt_2,51,165,282).  visible(bww1gt_2,282). happensAt(walking(bww1gt_2),282).
orientation(bww1gt_1,173,283). coords(bww1gt_1,95,258,283).  visible(bww1gt_1,283). happensAt(walking(bww1gt_1),283). orientation(bww1gt_2,167,283). coords(bww1gt_2,51,165,283).  visible(bww1gt_2,283). happensAt(walking(bww1gt_2),283).
orientation(bww1gt_1,173,284). coords(bww1gt_1,95,258,284).  visible(bww1gt_1,284). happensAt(walking(bww1gt_1),284). orientation(bww1gt_2,167,284). coords(bww1gt_2,51,165,284).  visible(bww1gt_2,284). happensAt(walking(bww1gt_2),284).
orientation(bww1gt_1,173,285). coords(bww1gt_1,95,259,285).  visible(bww1gt_1,285). happensAt(walking(bww1gt_1),285). orientation(bww1gt_2,167,285). coords(bww1gt_2,52,165,285).  visible(bww1gt_2,285). happensAt(walking(bww1gt_2),285).
orientation(bww1gt_1,173,286). coords(bww1gt_1,97,260,286).  visible(bww1gt_1,286). happensAt(walking(bww1gt_1),286). orientation(bww1gt_2,167,286). coords(bww1gt_2,52,165,286).  visible(bww1gt_2,286). happensAt(walking(bww1gt_2),286).
orientation(bww1gt_1,173,287). coords(bww1gt_1,99,260,287).  visible(bww1gt_1,287). happensAt(walking(bww1gt_1),287). orientation(bww1gt_2,167,287). coords(bww1gt_2,52,165,287).  visible(bww1gt_2,287). happensAt(walking(bww1gt_2),287).
orientation(bww1gt_1,172,288). coords(bww1gt_1,100,261,288).  visible(bww1gt_1,288). happensAt(walking(bww1gt_1),288). orientation(bww1gt_2,167,288). coords(bww1gt_2,53,164,288).  visible(bww1gt_2,288). happensAt(walking(bww1gt_2),288).
orientation(bww1gt_1,0,289). coords(bww1gt_1,101,261,289).  visible(bww1gt_1,289). happensAt(inactive(bww1gt_1),289). orientation(bww1gt_2,167,289). coords(bww1gt_2,53,164,289).  visible(bww1gt_2,289). happensAt(walking(bww1gt_2),289).
orientation(bww1gt_1,7,290). coords(bww1gt_1,101,262,290).  visible(bww1gt_1,290). happensAt(inactive(bww1gt_1),290). orientation(bww1gt_2,167,290). coords(bww1gt_2,54,164,290).  visible(bww1gt_2,290). happensAt(walking(bww1gt_2),290).
orientation(bww1gt_1,3,291). coords(bww1gt_1,101,262,291).  visible(bww1gt_1,291). happensAt(inactive(bww1gt_1),291). orientation(bww1gt_2,167,291). coords(bww1gt_2,55,164,291).  visible(bww1gt_2,291). happensAt(walking(bww1gt_2),291).
orientation(bww1gt_1,7,292). coords(bww1gt_1,102,262,292).  visible(bww1gt_1,292). happensAt(inactive(bww1gt_1),292). orientation(bww1gt_2,167,292). coords(bww1gt_2,56,164,292).  visible(bww1gt_2,292). happensAt(walking(bww1gt_2),292).
orientation(bww1gt_1,4,293). coords(bww1gt_1,102,262,293).  visible(bww1gt_1,293). happensAt(inactive(bww1gt_1),293). orientation(bww1gt_2,167,293). coords(bww1gt_2,56,164,293).  visible(bww1gt_2,293). happensAt(walking(bww1gt_2),293).
orientation(bww1gt_1,4,294). coords(bww1gt_1,102,262,294).  visible(bww1gt_1,294). happensAt(inactive(bww1gt_1),294). orientation(bww1gt_2,167,294). coords(bww1gt_2,56,164,294).  visible(bww1gt_2,294). happensAt(walking(bww1gt_2),294).
orientation(bww1gt_1,7,295). coords(bww1gt_1,103,263,295).  visible(bww1gt_1,295). happensAt(inactive(bww1gt_1),295). orientation(bww1gt_2,167,295). coords(bww1gt_2,57,164,295).  visible(bww1gt_2,295). happensAt(walking(bww1gt_2),295).
orientation(bww1gt_1,4,296). coords(bww1gt_1,103,263,296).  visible(bww1gt_1,296). happensAt(inactive(bww1gt_1),296). orientation(bww1gt_2,167,296). coords(bww1gt_2,57,163,296).  visible(bww1gt_2,296). happensAt(walking(bww1gt_2),296).
orientation(bww1gt_1,8,297). coords(bww1gt_1,103,263,297).  visible(bww1gt_1,297). happensAt(inactive(bww1gt_1),297). orientation(bww1gt_2,167,297). coords(bww1gt_2,57,163,297).  visible(bww1gt_2,297). happensAt(walking(bww1gt_2),297).
orientation(bww1gt_1,179,298). coords(bww1gt_1,103,263,298).  visible(bww1gt_1,298). happensAt(inactive(bww1gt_1),298). orientation(bww1gt_2,167,298). coords(bww1gt_2,57,163,298).  visible(bww1gt_2,298). happensAt(walking(bww1gt_2),298).
orientation(bww1gt_1,0,299). coords(bww1gt_1,103,263,299).  visible(bww1gt_1,299). happensAt(inactive(bww1gt_1),299). orientation(bww1gt_2,167,299). coords(bww1gt_2,57,163,299).  visible(bww1gt_2,299). happensAt(walking(bww1gt_2),299).
orientation(bww1gt_1,3,300). coords(bww1gt_1,103,263,300).  visible(bww1gt_1,300). happensAt(inactive(bww1gt_1),300). orientation(bww1gt_2,167,300). coords(bww1gt_2,57,162,300).  visible(bww1gt_2,300). happensAt(walking(bww1gt_2),300).
orientation(bww1gt_1,6,301). coords(bww1gt_1,104,264,301).  visible(bww1gt_1,301). happensAt(inactive(bww1gt_1),301). orientation(bww1gt_2,167,301). coords(bww1gt_2,57,162,301).  visible(bww1gt_2,301). happensAt(walking(bww1gt_2),301).
orientation(bww1gt_1,3,302). coords(bww1gt_1,104,264,302).  visible(bww1gt_1,302). happensAt(inactive(bww1gt_1),302). orientation(bww1gt_2,167,302). coords(bww1gt_2,57,162,302).  visible(bww1gt_2,302). happensAt(walking(bww1gt_2),302).
#show holdsAt/2.