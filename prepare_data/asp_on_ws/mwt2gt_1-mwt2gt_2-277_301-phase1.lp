
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

	orientation(mwt2gt_1,143,277). coords(mwt2gt_1,138,75,277).  visible(mwt2gt_1,277). happensAt(walking(mwt2gt_1),277). orientation(mwt2gt_2,145,277). coords(mwt2gt_2,171,117,277).  visible(mwt2gt_2,277). happensAt(walking(mwt2gt_2),277).
orientation(mwt2gt_1,143,278). coords(mwt2gt_1,138,75,278).  visible(mwt2gt_1,278). happensAt(walking(mwt2gt_1),278). orientation(mwt2gt_2,145,278). coords(mwt2gt_2,170,116,278).  visible(mwt2gt_2,278). happensAt(walking(mwt2gt_2),278).
orientation(mwt2gt_1,143,279). coords(mwt2gt_1,138,75,279).  visible(mwt2gt_1,279). happensAt(walking(mwt2gt_1),279). orientation(mwt2gt_2,145,279). coords(mwt2gt_2,169,115,279).  visible(mwt2gt_2,279). happensAt(walking(mwt2gt_2),279).
orientation(mwt2gt_1,143,280). coords(mwt2gt_1,138,75,280).  visible(mwt2gt_1,280). happensAt(walking(mwt2gt_1),280). orientation(mwt2gt_2,145,280). coords(mwt2gt_2,167,113,280).  visible(mwt2gt_2,280). happensAt(walking(mwt2gt_2),280).
orientation(mwt2gt_1,143,281). coords(mwt2gt_1,138,75,281).  visible(mwt2gt_1,281). happensAt(walking(mwt2gt_1),281). orientation(mwt2gt_2,145,281). coords(mwt2gt_2,164,112,281).  visible(mwt2gt_2,281). happensAt(walking(mwt2gt_2),281).
orientation(mwt2gt_1,143,282). coords(mwt2gt_1,138,75,282).  visible(mwt2gt_1,282). happensAt(walking(mwt2gt_1),282). orientation(mwt2gt_2,145,282). coords(mwt2gt_2,164,110,282).  visible(mwt2gt_2,282). happensAt(walking(mwt2gt_2),282).
orientation(mwt2gt_1,143,283). coords(mwt2gt_1,138,75,283).  visible(mwt2gt_1,283). happensAt(walking(mwt2gt_1),283). orientation(mwt2gt_2,145,283). coords(mwt2gt_2,163,108,283).  visible(mwt2gt_2,283). happensAt(walking(mwt2gt_2),283).
orientation(mwt2gt_1,143,284). coords(mwt2gt_1,138,75,284).  visible(mwt2gt_1,284). happensAt(inactive(mwt2gt_1),284). orientation(mwt2gt_2,145,284). coords(mwt2gt_2,163,107,284).  visible(mwt2gt_2,284). happensAt(walking(mwt2gt_2),284).
orientation(mwt2gt_1,143,285). coords(mwt2gt_1,138,75,285).  visible(mwt2gt_1,285). happensAt(inactive(mwt2gt_1),285). orientation(mwt2gt_2,145,285). coords(mwt2gt_2,162,106,285).  visible(mwt2gt_2,285). happensAt(walking(mwt2gt_2),285).
orientation(mwt2gt_1,143,286). coords(mwt2gt_1,138,75,286).  visible(mwt2gt_1,286). happensAt(inactive(mwt2gt_1),286). orientation(mwt2gt_2,145,286). coords(mwt2gt_2,162,106,286).  visible(mwt2gt_2,286). happensAt(walking(mwt2gt_2),286).
orientation(mwt2gt_1,143,287). coords(mwt2gt_1,138,75,287).  visible(mwt2gt_1,287). happensAt(inactive(mwt2gt_1),287). orientation(mwt2gt_2,145,287). coords(mwt2gt_2,162,106,287).  visible(mwt2gt_2,287). happensAt(walking(mwt2gt_2),287).
orientation(mwt2gt_1,143,288). coords(mwt2gt_1,138,75,288).  visible(mwt2gt_1,288). happensAt(inactive(mwt2gt_1),288). orientation(mwt2gt_2,145,288). coords(mwt2gt_2,161,106,288).  visible(mwt2gt_2,288). happensAt(walking(mwt2gt_2),288).
orientation(mwt2gt_1,143,289). coords(mwt2gt_1,138,75,289).  visible(mwt2gt_1,289). happensAt(inactive(mwt2gt_1),289). orientation(mwt2gt_2,145,289). coords(mwt2gt_2,161,106,289).  visible(mwt2gt_2,289). happensAt(walking(mwt2gt_2),289).
orientation(mwt2gt_1,143,290). coords(mwt2gt_1,138,75,290).  visible(mwt2gt_1,290). happensAt(inactive(mwt2gt_1),290). orientation(mwt2gt_2,145,290). coords(mwt2gt_2,160,106,290).  visible(mwt2gt_2,290). happensAt(walking(mwt2gt_2),290).
orientation(mwt2gt_1,143,291). coords(mwt2gt_1,138,75,291).  visible(mwt2gt_1,291). happensAt(inactive(mwt2gt_1),291). orientation(mwt2gt_2,145,291). coords(mwt2gt_2,159,104,291).  visible(mwt2gt_2,291). happensAt(walking(mwt2gt_2),291).
orientation(mwt2gt_1,143,292). coords(mwt2gt_1,138,75,292).  visible(mwt2gt_1,292). happensAt(inactive(mwt2gt_1),292). orientation(mwt2gt_2,145,292). coords(mwt2gt_2,158,103,292).  visible(mwt2gt_2,292). happensAt(walking(mwt2gt_2),292).
orientation(mwt2gt_1,143,293). coords(mwt2gt_1,138,75,293).  visible(mwt2gt_1,293). happensAt(inactive(mwt2gt_1),293). orientation(mwt2gt_2,145,293). coords(mwt2gt_2,157,102,293).  visible(mwt2gt_2,293). happensAt(walking(mwt2gt_2),293).
orientation(mwt2gt_1,143,294). coords(mwt2gt_1,138,75,294).  visible(mwt2gt_1,294). happensAt(inactive(mwt2gt_1),294). orientation(mwt2gt_2,145,294). coords(mwt2gt_2,155,101,294).  visible(mwt2gt_2,294). happensAt(walking(mwt2gt_2),294).
orientation(mwt2gt_1,143,295). coords(mwt2gt_1,138,75,295).  visible(mwt2gt_1,295). happensAt(inactive(mwt2gt_1),295). orientation(mwt2gt_2,145,295). coords(mwt2gt_2,153,100,295).  visible(mwt2gt_2,295). happensAt(walking(mwt2gt_2),295).
orientation(mwt2gt_1,143,296). coords(mwt2gt_1,138,75,296).  visible(mwt2gt_1,296). happensAt(inactive(mwt2gt_1),296). orientation(mwt2gt_2,145,296). coords(mwt2gt_2,153,100,296).  visible(mwt2gt_2,296). happensAt(walking(mwt2gt_2),296).
orientation(mwt2gt_1,143,297). coords(mwt2gt_1,138,75,297).  visible(mwt2gt_1,297). happensAt(inactive(mwt2gt_1),297). orientation(mwt2gt_2,145,297). coords(mwt2gt_2,151,100,297).  visible(mwt2gt_2,297). happensAt(walking(mwt2gt_2),297).
orientation(mwt2gt_1,143,298). coords(mwt2gt_1,138,75,298).  visible(mwt2gt_1,298). happensAt(inactive(mwt2gt_1),298). orientation(mwt2gt_2,145,298). coords(mwt2gt_2,151,99,298).  visible(mwt2gt_2,298). happensAt(walking(mwt2gt_2),298).
orientation(mwt2gt_1,143,299). coords(mwt2gt_1,138,75,299).  visible(mwt2gt_1,299). happensAt(inactive(mwt2gt_1),299). orientation(mwt2gt_2,145,299). coords(mwt2gt_2,151,99,299).  visible(mwt2gt_2,299). happensAt(walking(mwt2gt_2),299).
orientation(mwt2gt_1,143,300). coords(mwt2gt_1,138,75,300).  visible(mwt2gt_1,300). happensAt(inactive(mwt2gt_1),300). orientation(mwt2gt_2,145,300). coords(mwt2gt_2,151,99,300).  visible(mwt2gt_2,300). happensAt(walking(mwt2gt_2),300).
orientation(mwt2gt_1,143,301). coords(mwt2gt_1,138,75,301).  visible(mwt2gt_1,301). happensAt(inactive(mwt2gt_1),301). orientation(mwt2gt_2,145,301). coords(mwt2gt_2,151,98,301).  visible(mwt2gt_2,301). happensAt(walking(mwt2gt_2),301).
#show holdsAt/2.