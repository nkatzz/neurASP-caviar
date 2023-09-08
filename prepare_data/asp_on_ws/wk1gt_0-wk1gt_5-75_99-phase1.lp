
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

	orientation(wk1gt_0,136,75). coords(wk1gt_0,218,162,75).  visible(wk1gt_0,75). happensAt(walking(wk1gt_0),75). orientation(wk1gt_5,0,75). coords(wk1gt_5,27,204,75).  visible(wk1gt_5,75). happensAt(walking(wk1gt_5),75).
orientation(wk1gt_0,136,76). coords(wk1gt_0,217,160,76).  visible(wk1gt_0,76). happensAt(walking(wk1gt_0),76). orientation(wk1gt_5,0,76). coords(wk1gt_5,28,204,76).  visible(wk1gt_5,76). happensAt(walking(wk1gt_5),76).
orientation(wk1gt_0,136,77). coords(wk1gt_0,215,156,77).  visible(wk1gt_0,77). happensAt(walking(wk1gt_0),77). orientation(wk1gt_5,0,77). coords(wk1gt_5,28,203,77).  visible(wk1gt_5,77). happensAt(walking(wk1gt_5),77).
orientation(wk1gt_0,131,78). coords(wk1gt_0,213,154,78).  visible(wk1gt_0,78). happensAt(walking(wk1gt_0),78). orientation(wk1gt_5,0,78). coords(wk1gt_5,28,203,78).  visible(wk1gt_5,78). happensAt(walking(wk1gt_5),78).
orientation(wk1gt_0,131,79). coords(wk1gt_0,212,153,79).  visible(wk1gt_0,79). happensAt(walking(wk1gt_0),79). orientation(wk1gt_5,0,79). coords(wk1gt_5,28,202,79).  visible(wk1gt_5,79). happensAt(walking(wk1gt_5),79).
orientation(wk1gt_0,124,80). coords(wk1gt_0,211,152,80).  visible(wk1gt_0,80). happensAt(walking(wk1gt_0),80). orientation(wk1gt_5,0,80). coords(wk1gt_5,29,201,80).  visible(wk1gt_5,80). happensAt(walking(wk1gt_5),80).
orientation(wk1gt_0,120,81). coords(wk1gt_0,210,151,81).  visible(wk1gt_0,81). happensAt(walking(wk1gt_0),81). orientation(wk1gt_5,0,81). coords(wk1gt_5,29,201,81).  visible(wk1gt_5,81). happensAt(walking(wk1gt_5),81).
orientation(wk1gt_0,115,82). coords(wk1gt_0,209,150,82).  visible(wk1gt_0,82). happensAt(walking(wk1gt_0),82). orientation(wk1gt_5,0,82). coords(wk1gt_5,29,200,82).  visible(wk1gt_5,82). happensAt(walking(wk1gt_5),82).
orientation(wk1gt_0,119,83). coords(wk1gt_0,209,150,83).  visible(wk1gt_0,83). happensAt(walking(wk1gt_0),83). orientation(wk1gt_5,0,83). coords(wk1gt_5,29,198,83).  visible(wk1gt_5,83). happensAt(walking(wk1gt_5),83).
orientation(wk1gt_0,121,84). coords(wk1gt_0,208,146,84).  visible(wk1gt_0,84). happensAt(walking(wk1gt_0),84). orientation(wk1gt_5,0,84). coords(wk1gt_5,30,197,84).  visible(wk1gt_5,84). happensAt(walking(wk1gt_5),84).
orientation(wk1gt_0,120,85). coords(wk1gt_0,208,144,85).  visible(wk1gt_0,85). happensAt(walking(wk1gt_0),85). orientation(wk1gt_5,0,85). coords(wk1gt_5,30,197,85).  visible(wk1gt_5,85). happensAt(walking(wk1gt_5),85).
orientation(wk1gt_0,119,86). coords(wk1gt_0,207,142,86).  visible(wk1gt_0,86). happensAt(walking(wk1gt_0),86). orientation(wk1gt_5,0,86). coords(wk1gt_5,31,197,86).  visible(wk1gt_5,86). happensAt(walking(wk1gt_5),86).
orientation(wk1gt_0,123,87). coords(wk1gt_0,207,138,87).  visible(wk1gt_0,87). happensAt(walking(wk1gt_0),87). orientation(wk1gt_5,0,87). coords(wk1gt_5,31,197,87).  visible(wk1gt_5,87). happensAt(walking(wk1gt_5),87).
orientation(wk1gt_0,125,88). coords(wk1gt_0,207,136,88).  visible(wk1gt_0,88). happensAt(walking(wk1gt_0),88). orientation(wk1gt_5,0,88). coords(wk1gt_5,32,197,88).  visible(wk1gt_5,88). happensAt(walking(wk1gt_5),88).
orientation(wk1gt_0,130,89). coords(wk1gt_0,207,134,89).  visible(wk1gt_0,89). happensAt(walking(wk1gt_0),89). orientation(wk1gt_5,0,89). coords(wk1gt_5,32,196,89).  visible(wk1gt_5,89). happensAt(walking(wk1gt_5),89).
orientation(wk1gt_0,131,90). coords(wk1gt_0,207,131,90).  visible(wk1gt_0,90). happensAt(walking(wk1gt_0),90). orientation(wk1gt_5,0,90). coords(wk1gt_5,32,195,90).  visible(wk1gt_5,90). happensAt(walking(wk1gt_5),90).
orientation(wk1gt_0,131,91). coords(wk1gt_0,207,129,91).  visible(wk1gt_0,91). happensAt(walking(wk1gt_0),91). orientation(wk1gt_5,0,91). coords(wk1gt_5,32,195,91).  visible(wk1gt_5,91). happensAt(walking(wk1gt_5),91).
orientation(wk1gt_0,131,92). coords(wk1gt_0,206,128,92).  visible(wk1gt_0,92). happensAt(walking(wk1gt_0),92). orientation(wk1gt_5,0,92). coords(wk1gt_5,32,194,92).  visible(wk1gt_5,92). happensAt(walking(wk1gt_5),92).
orientation(wk1gt_0,131,93). coords(wk1gt_0,206,127,93).  visible(wk1gt_0,93). happensAt(walking(wk1gt_0),93). orientation(wk1gt_5,0,93). coords(wk1gt_5,32,194,93).  visible(wk1gt_5,93). happensAt(walking(wk1gt_5),93).
orientation(wk1gt_0,131,94). coords(wk1gt_0,206,127,94).  visible(wk1gt_0,94). happensAt(walking(wk1gt_0),94). orientation(wk1gt_5,0,94). coords(wk1gt_5,32,193,94).  visible(wk1gt_5,94). happensAt(walking(wk1gt_5),94).
orientation(wk1gt_0,131,95). coords(wk1gt_0,206,126,95).  visible(wk1gt_0,95). happensAt(walking(wk1gt_0),95). orientation(wk1gt_5,0,95). coords(wk1gt_5,32,192,95).  visible(wk1gt_5,95). happensAt(walking(wk1gt_5),95).
orientation(wk1gt_0,131,96). coords(wk1gt_0,206,126,96).  visible(wk1gt_0,96). happensAt(walking(wk1gt_0),96). orientation(wk1gt_5,0,96). coords(wk1gt_5,32,191,96).  visible(wk1gt_5,96). happensAt(walking(wk1gt_5),96).
orientation(wk1gt_0,131,97). coords(wk1gt_0,206,126,97).  visible(wk1gt_0,97). happensAt(walking(wk1gt_0),97). orientation(wk1gt_5,0,97). coords(wk1gt_5,34,191,97).  visible(wk1gt_5,97). happensAt(walking(wk1gt_5),97).
orientation(wk1gt_0,131,98). coords(wk1gt_0,206,126,98).  visible(wk1gt_0,98). happensAt(walking(wk1gt_0),98). orientation(wk1gt_5,0,98). coords(wk1gt_5,35,189,98).  visible(wk1gt_5,98). happensAt(walking(wk1gt_5),98).
orientation(wk1gt_0,131,99). coords(wk1gt_0,205,125,99).  visible(wk1gt_0,99). happensAt(walking(wk1gt_0),99). orientation(wk1gt_5,0,99). coords(wk1gt_5,35,188,99).  visible(wk1gt_5,99). happensAt(walking(wk1gt_5),99).
#show holdsAt/2.