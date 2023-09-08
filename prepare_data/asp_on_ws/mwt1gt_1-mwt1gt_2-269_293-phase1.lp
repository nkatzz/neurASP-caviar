
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

	orientation(mwt1gt_1,160,269). coords(mwt1gt_1,79,176,269).  visible(mwt1gt_1,269). happensAt(walking(mwt1gt_1),269). orientation(mwt1gt_2,130,269). coords(mwt1gt_2,215,51,269).  visible(mwt1gt_2,269). happensAt(walking(mwt1gt_2),269).
orientation(mwt1gt_1,160,270). coords(mwt1gt_1,79,176,270).  visible(mwt1gt_1,270). happensAt(walking(mwt1gt_1),270). orientation(mwt1gt_2,130,270). coords(mwt1gt_2,213,53,270).  visible(mwt1gt_2,270). happensAt(walking(mwt1gt_2),270).
orientation(mwt1gt_1,160,271). coords(mwt1gt_1,79,176,271).  visible(mwt1gt_1,271). happensAt(walking(mwt1gt_1),271). orientation(mwt1gt_2,130,271). coords(mwt1gt_2,211,54,271).  visible(mwt1gt_2,271). happensAt(walking(mwt1gt_2),271).
orientation(mwt1gt_1,160,272). coords(mwt1gt_1,79,176,272).  visible(mwt1gt_1,272). happensAt(walking(mwt1gt_1),272). orientation(mwt1gt_2,130,272). coords(mwt1gt_2,208,54,272).  visible(mwt1gt_2,272). happensAt(walking(mwt1gt_2),272).
orientation(mwt1gt_1,160,273). coords(mwt1gt_1,79,175,273).  visible(mwt1gt_1,273). happensAt(walking(mwt1gt_1),273). orientation(mwt1gt_2,130,273). coords(mwt1gt_2,204,54,273).  visible(mwt1gt_2,273). happensAt(walking(mwt1gt_2),273).
orientation(mwt1gt_1,160,274). coords(mwt1gt_1,81,175,274).  visible(mwt1gt_1,274). happensAt(walking(mwt1gt_1),274). orientation(mwt1gt_2,130,274). coords(mwt1gt_2,200,55,274).  visible(mwt1gt_2,274). happensAt(walking(mwt1gt_2),274).
orientation(mwt1gt_1,160,275). coords(mwt1gt_1,81,175,275).  visible(mwt1gt_1,275). happensAt(walking(mwt1gt_1),275). orientation(mwt1gt_2,130,275). coords(mwt1gt_2,199,56,275).  visible(mwt1gt_2,275). happensAt(walking(mwt1gt_2),275).
orientation(mwt1gt_1,160,276). coords(mwt1gt_1,83,175,276).  visible(mwt1gt_1,276). happensAt(walking(mwt1gt_1),276). orientation(mwt1gt_2,130,276). coords(mwt1gt_2,198,57,276).  visible(mwt1gt_2,276). happensAt(walking(mwt1gt_2),276).
orientation(mwt1gt_1,160,277). coords(mwt1gt_1,84,175,277).  visible(mwt1gt_1,277). happensAt(walking(mwt1gt_1),277). orientation(mwt1gt_2,130,277). coords(mwt1gt_2,197,60,277).  visible(mwt1gt_2,277). happensAt(walking(mwt1gt_2),277).
orientation(mwt1gt_1,160,278). coords(mwt1gt_1,84,175,278).  visible(mwt1gt_1,278). happensAt(walking(mwt1gt_1),278). orientation(mwt1gt_2,130,278). coords(mwt1gt_2,196,61,278).  visible(mwt1gt_2,278). happensAt(walking(mwt1gt_2),278).
orientation(mwt1gt_1,160,279). coords(mwt1gt_1,86,175,279).  visible(mwt1gt_1,279). happensAt(walking(mwt1gt_1),279). orientation(mwt1gt_2,130,279). coords(mwt1gt_2,195,64,279).  visible(mwt1gt_2,279). happensAt(walking(mwt1gt_2),279).
orientation(mwt1gt_1,160,280). coords(mwt1gt_1,87,175,280).  visible(mwt1gt_1,280). happensAt(walking(mwt1gt_1),280). orientation(mwt1gt_2,130,280). coords(mwt1gt_2,194,65,280).  visible(mwt1gt_2,280). happensAt(walking(mwt1gt_2),280).
orientation(mwt1gt_1,160,281). coords(mwt1gt_1,88,174,281).  visible(mwt1gt_1,281). happensAt(walking(mwt1gt_1),281). orientation(mwt1gt_2,130,281). coords(mwt1gt_2,193,67,281).  visible(mwt1gt_2,281). happensAt(walking(mwt1gt_2),281).
orientation(mwt1gt_1,160,282). coords(mwt1gt_1,89,174,282).  visible(mwt1gt_1,282). happensAt(walking(mwt1gt_1),282). orientation(mwt1gt_2,130,282). coords(mwt1gt_2,191,69,282).  visible(mwt1gt_2,282). happensAt(walking(mwt1gt_2),282).
orientation(mwt1gt_1,160,283). coords(mwt1gt_1,89,174,283).  visible(mwt1gt_1,283). happensAt(walking(mwt1gt_1),283). orientation(mwt1gt_2,130,283). coords(mwt1gt_2,191,69,283).  visible(mwt1gt_2,283). happensAt(walking(mwt1gt_2),283).
orientation(mwt1gt_1,160,284). coords(mwt1gt_1,89,173,284).  visible(mwt1gt_1,284). happensAt(walking(mwt1gt_1),284). orientation(mwt1gt_2,130,284). coords(mwt1gt_2,188,70,284).  visible(mwt1gt_2,284). happensAt(walking(mwt1gt_2),284).
orientation(mwt1gt_1,160,285). coords(mwt1gt_1,89,173,285).  visible(mwt1gt_1,285). happensAt(walking(mwt1gt_1),285). orientation(mwt1gt_2,130,285). coords(mwt1gt_2,185,70,285).  visible(mwt1gt_2,285). happensAt(walking(mwt1gt_2),285).
orientation(mwt1gt_1,160,286). coords(mwt1gt_1,90,171,286).  visible(mwt1gt_1,286). happensAt(walking(mwt1gt_1),286). orientation(mwt1gt_2,130,286). coords(mwt1gt_2,183,71,286).  visible(mwt1gt_2,286). happensAt(walking(mwt1gt_2),286).
orientation(mwt1gt_1,160,287). coords(mwt1gt_1,90,171,287).  visible(mwt1gt_1,287). happensAt(walking(mwt1gt_1),287). orientation(mwt1gt_2,130,287). coords(mwt1gt_2,180,72,287).  visible(mwt1gt_2,287). happensAt(walking(mwt1gt_2),287).
orientation(mwt1gt_1,160,288). coords(mwt1gt_1,90,169,288).  visible(mwt1gt_1,288). happensAt(walking(mwt1gt_1),288). orientation(mwt1gt_2,130,288). coords(mwt1gt_2,177,74,288).  visible(mwt1gt_2,288). happensAt(walking(mwt1gt_2),288).
orientation(mwt1gt_1,160,289). coords(mwt1gt_1,91,169,289).  visible(mwt1gt_1,289). happensAt(walking(mwt1gt_1),289). orientation(mwt1gt_2,130,289). coords(mwt1gt_2,176,77,289).  visible(mwt1gt_2,289). happensAt(walking(mwt1gt_2),289).
orientation(mwt1gt_1,160,290). coords(mwt1gt_1,91,168,290).  visible(mwt1gt_1,290). happensAt(walking(mwt1gt_1),290). orientation(mwt1gt_2,130,290). coords(mwt1gt_2,175,79,290).  visible(mwt1gt_2,290). happensAt(walking(mwt1gt_2),290).
orientation(mwt1gt_1,160,291). coords(mwt1gt_1,94,168,291).  visible(mwt1gt_1,291). happensAt(walking(mwt1gt_1),291). orientation(mwt1gt_2,130,291). coords(mwt1gt_2,174,82,291).  visible(mwt1gt_2,291). happensAt(walking(mwt1gt_2),291).
orientation(mwt1gt_1,160,292). coords(mwt1gt_1,96,168,292).  visible(mwt1gt_1,292). happensAt(walking(mwt1gt_1),292). orientation(mwt1gt_2,130,292). coords(mwt1gt_2,173,84,292).  visible(mwt1gt_2,292). happensAt(walking(mwt1gt_2),292).
orientation(mwt1gt_1,160,293). coords(mwt1gt_1,97,167,293).  visible(mwt1gt_1,293). happensAt(walking(mwt1gt_1),293). orientation(mwt1gt_2,130,293). coords(mwt1gt_2,172,86,293).  visible(mwt1gt_2,293). happensAt(walking(mwt1gt_2),293).
#show holdsAt/2.