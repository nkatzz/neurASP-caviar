
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

	orientation(mc1gt_0,148,81). coords(mc1gt_0,79,96,81).  visible(mc1gt_0,81). happensAt(walking(mc1gt_0),81). orientation(mc1gt_2,144,81). coords(mc1gt_2,78,65,81).  visible(mc1gt_2,81). happensAt(walking(mc1gt_2),81).
orientation(mc1gt_0,148,82). coords(mc1gt_0,79,96,82).  visible(mc1gt_0,82). happensAt(walking(mc1gt_0),82). orientation(mc1gt_2,144,82). coords(mc1gt_2,78,65,82).  visible(mc1gt_2,82). happensAt(walking(mc1gt_2),82).
orientation(mc1gt_0,148,83). coords(mc1gt_0,79,96,83).  visible(mc1gt_0,83). happensAt(walking(mc1gt_0),83). orientation(mc1gt_2,144,83). coords(mc1gt_2,78,65,83).  visible(mc1gt_2,83). happensAt(walking(mc1gt_2),83).
orientation(mc1gt_0,148,84). coords(mc1gt_0,79,96,84).  visible(mc1gt_0,84). happensAt(walking(mc1gt_0),84). orientation(mc1gt_2,144,84). coords(mc1gt_2,78,65,84).  visible(mc1gt_2,84). happensAt(walking(mc1gt_2),84).
orientation(mc1gt_0,148,85). coords(mc1gt_0,79,96,85).  visible(mc1gt_0,85). happensAt(walking(mc1gt_0),85). orientation(mc1gt_2,144,85). coords(mc1gt_2,78,65,85).  visible(mc1gt_2,85). happensAt(walking(mc1gt_2),85).
orientation(mc1gt_0,148,86). coords(mc1gt_0,79,96,86).  visible(mc1gt_0,86). happensAt(walking(mc1gt_0),86). orientation(mc1gt_2,144,86). coords(mc1gt_2,78,65,86).  visible(mc1gt_2,86). happensAt(walking(mc1gt_2),86).
orientation(mc1gt_0,148,87). coords(mc1gt_0,79,96,87).  visible(mc1gt_0,87). happensAt(walking(mc1gt_0),87). orientation(mc1gt_2,144,87). coords(mc1gt_2,78,65,87).  visible(mc1gt_2,87). happensAt(walking(mc1gt_2),87).
orientation(mc1gt_0,148,88). coords(mc1gt_0,79,96,88).  visible(mc1gt_0,88). happensAt(walking(mc1gt_0),88). orientation(mc1gt_2,144,88). coords(mc1gt_2,78,65,88).  visible(mc1gt_2,88). happensAt(walking(mc1gt_2),88).
orientation(mc1gt_0,148,89). coords(mc1gt_0,79,96,89).  visible(mc1gt_0,89). happensAt(walking(mc1gt_0),89). orientation(mc1gt_2,144,89). coords(mc1gt_2,78,65,89).  visible(mc1gt_2,89). happensAt(walking(mc1gt_2),89).
orientation(mc1gt_0,148,90). coords(mc1gt_0,79,96,90).  visible(mc1gt_0,90). happensAt(walking(mc1gt_0),90). orientation(mc1gt_2,144,90). coords(mc1gt_2,78,65,90).  visible(mc1gt_2,90). happensAt(walking(mc1gt_2),90).
orientation(mc1gt_0,148,91). coords(mc1gt_0,79,96,91).  visible(mc1gt_0,91). happensAt(walking(mc1gt_0),91). orientation(mc1gt_2,144,91). coords(mc1gt_2,78,65,91).  visible(mc1gt_2,91). happensAt(walking(mc1gt_2),91).
orientation(mc1gt_0,148,92). coords(mc1gt_0,79,96,92).  visible(mc1gt_0,92). happensAt(walking(mc1gt_0),92). orientation(mc1gt_2,144,92). coords(mc1gt_2,78,65,92).  visible(mc1gt_2,92). happensAt(walking(mc1gt_2),92).
orientation(mc1gt_0,148,93). coords(mc1gt_0,79,96,93).  visible(mc1gt_0,93). happensAt(walking(mc1gt_0),93). orientation(mc1gt_2,144,93). coords(mc1gt_2,78,65,93).  visible(mc1gt_2,93). happensAt(walking(mc1gt_2),93).
orientation(mc1gt_0,148,94). coords(mc1gt_0,79,96,94).  visible(mc1gt_0,94). happensAt(walking(mc1gt_0),94). orientation(mc1gt_2,144,94). coords(mc1gt_2,78,65,94).  visible(mc1gt_2,94). happensAt(walking(mc1gt_2),94).
orientation(mc1gt_0,148,95). coords(mc1gt_0,79,96,95).  visible(mc1gt_0,95). happensAt(walking(mc1gt_0),95). orientation(mc1gt_2,144,95). coords(mc1gt_2,78,65,95).  visible(mc1gt_2,95). happensAt(walking(mc1gt_2),95).
orientation(mc1gt_0,148,96). coords(mc1gt_0,79,96,96).  visible(mc1gt_0,96). happensAt(walking(mc1gt_0),96). orientation(mc1gt_2,144,96). coords(mc1gt_2,78,65,96).  visible(mc1gt_2,96). happensAt(walking(mc1gt_2),96).
orientation(mc1gt_0,148,97). coords(mc1gt_0,79,96,97).  visible(mc1gt_0,97). happensAt(walking(mc1gt_0),97). orientation(mc1gt_2,144,97). coords(mc1gt_2,78,65,97).  visible(mc1gt_2,97). happensAt(walking(mc1gt_2),97).
orientation(mc1gt_0,148,98). coords(mc1gt_0,79,96,98).  visible(mc1gt_0,98). happensAt(walking(mc1gt_0),98). orientation(mc1gt_2,144,98). coords(mc1gt_2,78,65,98).  visible(mc1gt_2,98). happensAt(walking(mc1gt_2),98).
orientation(mc1gt_0,148,99). coords(mc1gt_0,79,96,99).  visible(mc1gt_0,99). happensAt(walking(mc1gt_0),99). orientation(mc1gt_2,144,99). coords(mc1gt_2,78,65,99).  visible(mc1gt_2,99). happensAt(walking(mc1gt_2),99).
orientation(mc1gt_0,148,100). coords(mc1gt_0,79,96,100).  visible(mc1gt_0,100). happensAt(walking(mc1gt_0),100). orientation(mc1gt_2,144,100). coords(mc1gt_2,78,65,100).  visible(mc1gt_2,100). happensAt(walking(mc1gt_2),100).
orientation(mc1gt_0,148,101). coords(mc1gt_0,79,97,101).  visible(mc1gt_0,101). happensAt(walking(mc1gt_0),101). orientation(mc1gt_2,144,101). coords(mc1gt_2,77,66,101).  visible(mc1gt_2,101). happensAt(walking(mc1gt_2),101).
orientation(mc1gt_0,148,102). coords(mc1gt_0,79,97,102).  visible(mc1gt_0,102). happensAt(walking(mc1gt_0),102). orientation(mc1gt_2,144,102). coords(mc1gt_2,77,66,102).  visible(mc1gt_2,102). happensAt(walking(mc1gt_2),102).
orientation(mc1gt_0,148,103). coords(mc1gt_0,79,97,103).  visible(mc1gt_0,103). happensAt(walking(mc1gt_0),103). orientation(mc1gt_2,144,103). coords(mc1gt_2,77,67,103).  visible(mc1gt_2,103). happensAt(walking(mc1gt_2),103).
orientation(mc1gt_0,148,104). coords(mc1gt_0,80,97,104).  visible(mc1gt_0,104). happensAt(walking(mc1gt_0),104). orientation(mc1gt_2,144,104). coords(mc1gt_2,76,67,104).  visible(mc1gt_2,104). happensAt(walking(mc1gt_2),104).
orientation(mc1gt_0,148,105). coords(mc1gt_0,80,97,105).  visible(mc1gt_0,105). happensAt(walking(mc1gt_0),105). orientation(mc1gt_2,144,105). coords(mc1gt_2,76,68,105).  visible(mc1gt_2,105). happensAt(walking(mc1gt_2),105).
#show holdsAt/2.