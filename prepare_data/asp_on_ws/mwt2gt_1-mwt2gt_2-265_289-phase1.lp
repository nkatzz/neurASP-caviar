
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

	orientation(mwt2gt_1,143,265). coords(mwt2gt_1,138,75,265).  visible(mwt2gt_1,265). happensAt(walking(mwt2gt_1),265). orientation(mwt2gt_2,145,265). coords(mwt2gt_2,184,128,265).  visible(mwt2gt_2,265). happensAt(walking(mwt2gt_2),265).
orientation(mwt2gt_1,143,266). coords(mwt2gt_1,138,75,266).  visible(mwt2gt_1,266). happensAt(walking(mwt2gt_1),266). orientation(mwt2gt_2,145,266). coords(mwt2gt_2,183,126,266).  visible(mwt2gt_2,266). happensAt(walking(mwt2gt_2),266).
orientation(mwt2gt_1,143,267). coords(mwt2gt_1,138,75,267).  visible(mwt2gt_1,267). happensAt(walking(mwt2gt_1),267). orientation(mwt2gt_2,145,267). coords(mwt2gt_2,182,125,267).  visible(mwt2gt_2,267). happensAt(walking(mwt2gt_2),267).
orientation(mwt2gt_1,143,268). coords(mwt2gt_1,138,75,268).  visible(mwt2gt_1,268). happensAt(walking(mwt2gt_1),268). orientation(mwt2gt_2,145,268). coords(mwt2gt_2,179,123,268).  visible(mwt2gt_2,268). happensAt(walking(mwt2gt_2),268).
orientation(mwt2gt_1,143,269). coords(mwt2gt_1,138,75,269).  visible(mwt2gt_1,269). happensAt(walking(mwt2gt_1),269). orientation(mwt2gt_2,145,269). coords(mwt2gt_2,177,121,269).  visible(mwt2gt_2,269). happensAt(walking(mwt2gt_2),269).
orientation(mwt2gt_1,143,270). coords(mwt2gt_1,138,75,270).  visible(mwt2gt_1,270). happensAt(walking(mwt2gt_1),270). orientation(mwt2gt_2,145,270). coords(mwt2gt_2,176,120,270).  visible(mwt2gt_2,270). happensAt(walking(mwt2gt_2),270).
orientation(mwt2gt_1,143,271). coords(mwt2gt_1,138,75,271).  visible(mwt2gt_1,271). happensAt(walking(mwt2gt_1),271). orientation(mwt2gt_2,145,271). coords(mwt2gt_2,176,120,271).  visible(mwt2gt_2,271). happensAt(walking(mwt2gt_2),271).
orientation(mwt2gt_1,143,272). coords(mwt2gt_1,138,75,272).  visible(mwt2gt_1,272). happensAt(walking(mwt2gt_1),272). orientation(mwt2gt_2,145,272). coords(mwt2gt_2,174,120,272).  visible(mwt2gt_2,272). happensAt(walking(mwt2gt_2),272).
orientation(mwt2gt_1,143,273). coords(mwt2gt_1,138,75,273).  visible(mwt2gt_1,273). happensAt(walking(mwt2gt_1),273). orientation(mwt2gt_2,145,273). coords(mwt2gt_2,174,120,273).  visible(mwt2gt_2,273). happensAt(walking(mwt2gt_2),273).
orientation(mwt2gt_1,143,274). coords(mwt2gt_1,138,75,274).  visible(mwt2gt_1,274). happensAt(walking(mwt2gt_1),274). orientation(mwt2gt_2,145,274). coords(mwt2gt_2,173,119,274).  visible(mwt2gt_2,274). happensAt(walking(mwt2gt_2),274).
orientation(mwt2gt_1,143,275). coords(mwt2gt_1,138,75,275).  visible(mwt2gt_1,275). happensAt(walking(mwt2gt_1),275). orientation(mwt2gt_2,145,275). coords(mwt2gt_2,173,119,275).  visible(mwt2gt_2,275). happensAt(walking(mwt2gt_2),275).
orientation(mwt2gt_1,143,276). coords(mwt2gt_1,138,75,276).  visible(mwt2gt_1,276). happensAt(walking(mwt2gt_1),276). orientation(mwt2gt_2,145,276). coords(mwt2gt_2,172,118,276).  visible(mwt2gt_2,276). happensAt(walking(mwt2gt_2),276).
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
#show holdsAt/2.