
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

	orientation(mws1gt_2,162,80). coords(mws1gt_2,36,173,80).  visible(mws1gt_2,80). happensAt(walking(mws1gt_2),80). orientation(mws1gt_3,27,80). coords(mws1gt_3,258,268,80).  visible(mws1gt_3,80). happensAt(walking(mws1gt_3),80).
orientation(mws1gt_2,162,81). coords(mws1gt_2,35,174,81).  visible(mws1gt_2,81). happensAt(walking(mws1gt_2),81). orientation(mws1gt_3,29,81). coords(mws1gt_3,257,267,81).  visible(mws1gt_3,81). happensAt(walking(mws1gt_3),81).
orientation(mws1gt_2,162,82). coords(mws1gt_2,35,175,82).  visible(mws1gt_2,82). happensAt(walking(mws1gt_2),82). orientation(mws1gt_3,29,82). coords(mws1gt_3,256,265,82).  visible(mws1gt_3,82). happensAt(walking(mws1gt_3),82).
orientation(mws1gt_2,162,83). coords(mws1gt_2,35,176,83).  visible(mws1gt_2,83). happensAt(walking(mws1gt_2),83). orientation(mws1gt_3,26,83). coords(mws1gt_3,254,263,83).  visible(mws1gt_3,83). happensAt(walking(mws1gt_3),83).
orientation(mws1gt_2,162,84). coords(mws1gt_2,35,177,84).  visible(mws1gt_2,84). happensAt(walking(mws1gt_2),84). orientation(mws1gt_3,24,84). coords(mws1gt_3,254,261,84).  visible(mws1gt_3,84). happensAt(walking(mws1gt_3),84).
orientation(mws1gt_2,162,85). coords(mws1gt_2,35,178,85).  visible(mws1gt_2,85). happensAt(walking(mws1gt_2),85). orientation(mws1gt_3,24,85). coords(mws1gt_3,252,256,85).  visible(mws1gt_3,85). happensAt(walking(mws1gt_3),85).
orientation(mws1gt_2,162,86). coords(mws1gt_2,35,179,86).  visible(mws1gt_2,86). happensAt(walking(mws1gt_2),86). orientation(mws1gt_3,24,86). coords(mws1gt_3,250,252,86).  visible(mws1gt_3,86). happensAt(walking(mws1gt_3),86).
orientation(mws1gt_2,162,87). coords(mws1gt_2,35,181,87).  visible(mws1gt_2,87). happensAt(walking(mws1gt_2),87). orientation(mws1gt_3,24,87). coords(mws1gt_3,250,249,87).  visible(mws1gt_3,87). happensAt(walking(mws1gt_3),87).
orientation(mws1gt_2,162,88). coords(mws1gt_2,35,181,88).  visible(mws1gt_2,88). happensAt(walking(mws1gt_2),88). orientation(mws1gt_3,24,88). coords(mws1gt_3,250,246,88).  visible(mws1gt_3,88). happensAt(walking(mws1gt_3),88).
orientation(mws1gt_2,162,89). coords(mws1gt_2,35,181,89).  visible(mws1gt_2,89). happensAt(walking(mws1gt_2),89). orientation(mws1gt_3,24,89). coords(mws1gt_3,250,245,89).  visible(mws1gt_3,89). happensAt(walking(mws1gt_3),89).
orientation(mws1gt_2,162,90). coords(mws1gt_2,34,181,90).  visible(mws1gt_2,90). happensAt(walking(mws1gt_2),90). orientation(mws1gt_3,24,90). coords(mws1gt_3,249,245,90).  visible(mws1gt_3,90). happensAt(walking(mws1gt_3),90).
orientation(mws1gt_2,162,91). coords(mws1gt_2,34,182,91).  visible(mws1gt_2,91). happensAt(walking(mws1gt_2),91). orientation(mws1gt_3,24,91). coords(mws1gt_3,248,245,91).  visible(mws1gt_3,91). happensAt(walking(mws1gt_3),91).
orientation(mws1gt_2,162,92). coords(mws1gt_2,34,182,92).  visible(mws1gt_2,92). happensAt(walking(mws1gt_2),92). orientation(mws1gt_3,24,92). coords(mws1gt_3,248,244,92).  visible(mws1gt_3,92). happensAt(walking(mws1gt_3),92).
orientation(mws1gt_2,162,93). coords(mws1gt_2,34,183,93).  visible(mws1gt_2,93). happensAt(walking(mws1gt_2),93). orientation(mws1gt_3,24,93). coords(mws1gt_3,246,243,93).  visible(mws1gt_3,93). happensAt(walking(mws1gt_3),93).
orientation(mws1gt_2,162,94). coords(mws1gt_2,34,185,94).  visible(mws1gt_2,94). happensAt(walking(mws1gt_2),94). orientation(mws1gt_3,24,94). coords(mws1gt_3,246,241,94).  visible(mws1gt_3,94). happensAt(walking(mws1gt_3),94).
orientation(mws1gt_2,162,95). coords(mws1gt_2,34,185,95).  visible(mws1gt_2,95). happensAt(walking(mws1gt_2),95). orientation(mws1gt_3,19,95). coords(mws1gt_3,244,240,95).  visible(mws1gt_3,95). happensAt(walking(mws1gt_3),95).
orientation(mws1gt_2,162,96). coords(mws1gt_2,34,187,96).  visible(mws1gt_2,96). happensAt(walking(mws1gt_2),96). orientation(mws1gt_3,10,96). coords(mws1gt_3,242,237,96).  visible(mws1gt_3,96). happensAt(walking(mws1gt_3),96).
orientation(mws1gt_2,162,97). coords(mws1gt_2,34,188,97).  visible(mws1gt_2,97). happensAt(walking(mws1gt_2),97). orientation(mws1gt_3,10,97). coords(mws1gt_3,242,236,97).  visible(mws1gt_3,97). happensAt(walking(mws1gt_3),97).
orientation(mws1gt_2,162,98). coords(mws1gt_2,34,189,98).  visible(mws1gt_2,98). happensAt(walking(mws1gt_2),98). orientation(mws1gt_3,11,98). coords(mws1gt_3,242,234,98).  visible(mws1gt_3,98). happensAt(walking(mws1gt_3),98).
orientation(mws1gt_2,162,99). coords(mws1gt_2,34,190,99).  visible(mws1gt_2,99). happensAt(walking(mws1gt_2),99). orientation(mws1gt_3,6,99). coords(mws1gt_3,241,231,99).  visible(mws1gt_3,99). happensAt(walking(mws1gt_3),99).
orientation(mws1gt_2,0,100). coords(mws1gt_2,34,189,100).  visible(mws1gt_2,100). happensAt(walking(mws1gt_2),100). orientation(mws1gt_3,3,100). coords(mws1gt_3,241,229,100).  visible(mws1gt_3,100). happensAt(walking(mws1gt_3),100).
orientation(mws1gt_2,0,101). coords(mws1gt_2,34,190,101).  visible(mws1gt_2,101). happensAt(walking(mws1gt_2),101). orientation(mws1gt_3,171,101). coords(mws1gt_3,240,226,101).  visible(mws1gt_3,101). happensAt(walking(mws1gt_3),101).
orientation(mws1gt_2,0,102). coords(mws1gt_2,34,191,102).  visible(mws1gt_2,102). happensAt(walking(mws1gt_2),102). orientation(mws1gt_3,4,102). coords(mws1gt_3,239,223,102).  visible(mws1gt_3,102). happensAt(walking(mws1gt_3),102).
orientation(mws1gt_2,167,103). coords(mws1gt_2,34,192,103).  visible(mws1gt_2,103). happensAt(walking(mws1gt_2),103). orientation(mws1gt_3,4,103). coords(mws1gt_3,239,222,103).  visible(mws1gt_3,103). happensAt(walking(mws1gt_3),103).
orientation(mws1gt_2,167,104). coords(mws1gt_2,34,193,104).  visible(mws1gt_2,104). happensAt(walking(mws1gt_2),104). orientation(mws1gt_3,174,104). coords(mws1gt_3,238,221,104).  visible(mws1gt_3,104). happensAt(walking(mws1gt_3),104).
#show holdsAt/2.